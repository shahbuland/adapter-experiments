from typing import List, Union
from dataclasses import dataclass, field

import torch
from torch import nn

from .utils import Modality, freeze_module
from .nn.blocks import StackedTransformer

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
    abstractor_kwargs : Dict = field(default_factory = lambda x : {
        "n_layers" : 4,
        "n_heads" : 8,
        "dim" : 256,
        "flash" : False
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

    if name == "detr":
        hidden_states = output.decoder_hidden_states
    else:
        hidden_states = output.hidden_states

    return hidden_states[-skip_index]


class MergedAdapter(nn.Module):
    """
    Multiple multimodal adapters merged together.
    Adapters are assumed to have a ViT backend with output_hidden_states as an option
    """
    def __init__(self, adapters, config : MergedAdapterConfig):
        super().__init__()

        self.adapters = nn.ModuleList([adapters])
        self.config = config

        if config.freeze_adapters:
            for adapter in self.adapters:
                freeze_module(adapter)
        
        if config.use_abstractor:
            abs_dim = config.abstractor_kwargs['dim']
            self.proj_list = nn.ModuleList(
                [nn.Linear(adapter.config.hidden_size, abs_dim) for adapter in self.adapters]
            )
            self.final_proj = nn.Linear(abs_dim, config.final_dim)
            self.abstractor = StackedTransformer(**config.abstractor_kwargs)
        else:
            self.proj_list = nn.ModuleList(
                [nn.Linear(adapter.config.hidden_size, config.final_dim) for adapter in self.adapters]
            )
            self.final_proj = None

        if config.use_type_embeddings:
            n = len(self.adapters)
            d = config.abstractor_kwargs['dim'] if config.use_abstractor else config.final_dim
            self.type_embeddings = nn.Parameter(torch.randn(n, d) * 1e-6)
        else:
            self.type_embeddings = None
        
        if isinstance(config.layer_skips, list):
            self.skips = config.layer_skips
        else:
            self.skips = [config.layer_skips] * len(self.adapters)

        assert config.adapter_names or config.adapter_modalities, "In adapter config, must provide one of adapter_names or adapter_modalities (none provided)"
        assert not (config.adapter_names and config.adapter_modalities), "In adapter config, must provide ONLY one of adapter_names or adapter_modalities (both provided)"
    
    def forward(self, processor_out):
        # If adapter modalitities is true, we use those as keys for the processor output, otherwise we use adapter names as keys
        use_modality_keys = self.config.adapter_modalities is not None

        if use_modality_keys:
            keys = [modality.value for modality in Modality]
        else:
            keys = self.config.adapter_names

        adapter_inputs = [processor_out[key] for key in keys]
        # Hidden states from all the adapters
        model_outs = [extract_hidden(name, adapter(**inputs, output_hidden_states = True), skip) for (inputs, adapter, skip, name) in zip(adapter_inputs, self.adapters, self.skips, self.config.adapter_names)]
        
        if self.config.use_abstractor:
            # Projected to same dim as abstractor, they're all [B, N_i, D] now
            projected_out = [proj(model_out) for (proj, model_out) in zip(self.proj_list, model_outs)]
        else:
            projected_out = [proj(model_out) for (proj, model_out) in zip(self.proj_list, model_outs)]

        if self.config.use_type_embeddings:
            for i in range(len(projected_out)):
                b, n, d = projected_out[i].shape
                type_embed = eo.repeat(self.type_embeddings[i], 'd -> b n d', b = b, n = n)
                projected_out[i] + type_embed
        
        adapters_out = torch.cat(projected_out, dim = 1) # [B, N, D]
        if self.config.use_abstractor:
            adapters_out = self.abstractor(adapters_out)
            adapters_out = self.final_proj(adapters_out)

        return adapters_out


        
