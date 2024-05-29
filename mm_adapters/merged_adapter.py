from typing import List, Union, Dict
from dataclasses import dataclass, field

import torch
from torch import nn
import einops as eo

from .utils import Modality, freeze_module
from .nn.pooling import PerceiverPooling

@dataclass
class MergedAdapterConfig:
    """
    Config for merged adapter

    :param adapter_names: Model names for adapters to pair them with processors
    :param adapter_modalities: As an alternative to providing names, this links processors based purely on modality
    :param layer_skips: If single int i, return i-th last layer from hidden states of each adapter. If list for each adapter j, individual layer skips for each model
    :param freeze_adapters: Should adapters themselves be frozen and not trained?

    :param use_type_embeddings: Use embeddings for different modality types?

    :param use_abstractor: Use an abstractor to merge modalities before feeding to LLM?
    :param abstractor_kwargs: args for the abstractor transformer model

    :param final_dim: Hidden size for the LLM we are feeding to
    """
    adapter_names : List[str] = None
    adapter_modalities : List[Modality] = None
    layer_skips : Union[int, List[int]] = -2
    freeze_adapters : bool = True

    use_type_embeddings : bool = False

    use_abstractor : bool = False
    abstractor_kwargs : Dict = field(default_factory = lambda : {
        "n_layers" : 4,
        "n_heads" : 8,
        "hidden_size" : 256,
        "out_seq_len" : 256
    })
    final_dim : int = 768

def extract_hidden(name, output, skip_index):
    """
    Different models sometimes have different ways to get their hidden states.
    This deals with that cleanly for us

    :param name: Name identifier for model
    :param output: Entire output from model (as dict)
    :param skip_index: Skip last layers (i.e. if skip_index is 2, return 2nd last hidden state)
    """

    if hasattr(output, 'encoder_hidden_states'):
        hidden_states = output.encoder_hidden_states
    else:
        hidden_states = output.hidden_states
    hidden_states = hidden_states[skip_index]

    if name == "clap":
        # CLAP is weird and reshapes the hidden states internally
        # We have to reshape them back
        hidden_states = eo.rearrange(hidden_states, 'b d h w -> b (h w) d')

    if len(hidden_states.shape) > 3: # [B, ...,N, D]
        hidden_states = hidden_states.flatten(start_dim = 1, end_dim = -2)

    return hidden_states

def infer_hidden_size(model):
    # Try a variety of strategies

    res = None
    if hasattr(model, 'hidden_size'):
        res = model.hidden_size
    elif hasattr(model, 'config'):
        if hasattr(model.config, 'hidden_size'):
            res = model.config.hidden_size
        elif hasattr(model.config, 'vision_config'):
            if hasattr(model.config.vision_config, 'hidden_size'):
                res = model.config.vision_config.hidden_size
        elif hasattr(model.config, 'audio_config'):
            if hasattr(model.config.audio_config, 'hidden_size'):
                res = model.config.audio_config.hidden_size
        elif hasattr(model.config, 'encoder'):
            if hasattr(model.config.encoder, 'hidden_size'):
                res = model.config.encoder.hidden_size

    if res is not None:
        return res
        
    raise ValueError(f"Couldn't figure out {model.__class__.__name__} hidden size")

class MergedAdapter(nn.Module):
    """
    Multiple multimodal adapters merged together.
    Adapters are assumed to have a ViT backend with output_hidden_states as an option
    """
    def __init__(self, adapters, config : MergedAdapterConfig):
        super().__init__()

        self.adapters = nn.ModuleList(adapters)
        self.config = config

        if config.freeze_adapters:
            for adapter in self.adapters:
                freeze_module(adapter)
        
        if config.use_abstractor:
            abs_dim = config.abstractor_kwargs['hidden_size']
            self.proj_list = nn.ModuleList(
                [nn.Linear(infer_hidden_size(adapter), abs_dim) for adapter in self.adapters]
            )
            self.final_proj = nn.Linear(abs_dim, config.final_dim)
            self.abstractor = PerceiverPooling(**config.abstractor_kwargs)
        else:
            self.proj_list = nn.ModuleList(
                [nn.Linear(infer_hidden_size(adapter), config.final_dim) for adapter in self.adapters]
            )
            self.final_proj = None

        if config.use_type_embeddings:
            n = len(self.adapters)
            d = config.abstractor_kwargs['hidden_size'] if config.use_abstractor else config.final_dim
            self.type_embeddings = nn.Parameter(torch.randn(n, d) * 1e-6)
        else:
            self.type_embeddings = None
        
        if isinstance(config.layer_skips, list):
            self.skips = config.layer_skips
        else:
            self.skips = [config.layer_skips] * len(self.adapters)

        assert config.adapter_names or config.adapter_modalities, "In adapter config, must provide one of adapter_names or adapter_modalities (none provided)"
        assert not (config.adapter_names and config.adapter_modalities), "In adapter config, must provide ONLY one of adapter_names or adapter_modalities (both provided)"
    
    def forward(self, processor_out, drop_adapters : List[str] = []):
        """
        Forward call for merged adapter

        :param processor_out: Direct output from processor
        :param drop_modalities: Optional parameter. Keys/names for adapters we want to drop. This assumes we aren't using modality keys
        """
        # If adapter modalitities is true, we use those as keys for the processor output, otherwise we use adapter names as keys
        use_modality_keys = self.config.adapter_modalities is not None

        if use_modality_keys:
            keys = [modality.value for modality in Modality]
            selected_indices = list(range(len(self.adapters))) # TODO: use_modality_keys probably just overall isn't a good idea
        else:
            keys = self.config.adapter_names # names
            selected_indices = [keys.index(key) for key in keys if not key in drop_adapters] # Indices of selected 

        adapter_inputs = [processor_out[key] for key in keys]
        # Hidden states from all the adapters
        model_outs = [
            extract_hidden(keys[idx], self.adapters[idx](**adapter_inputs[idx], output_hidden_states = True), self.skips[idx]) \
            for idx in selected_indices
        ]

        if self.config.use_abstractor:
            # Projected to same dim as abstractor, they're all [B, N_i, D] now
            projected_out = [proj(model_out) for (proj, model_out) in zip(self.proj_list, model_outs)]
        else:
            projected_out = [proj(model_out) for (proj, model_out) in zip(self.proj_list, model_outs)]

        if self.config.use_type_embeddings:
            for i, adapter_idx in enumerate(selected_indices):
                b, n, d = projected_out[i].shape
                type_embed = eo.repeat(self.type_embeddings[adapter_idx], 'd -> b n d', b = b, n = n)
                projected_out[i] += type_embed
        
        adapters_out = torch.cat(projected_out, dim = 1) # [B, N, D]
        if self.config.use_abstractor:
            adapters_out = self.abstractor(adapters_out)
            adapters_out = self.final_proj(adapters_out)

        return adapters_out


        
