from dataclasses import dataclass
from typing import Dict, List, Iterable, Any

from PIL import Image

from .utils import Modality

@dataclass
class MergedProcessorConfig:
    """
    :param processor_ids: Name identifiers for models. Can be model ids from hub.
    :param processor_modalities: Modalities for processors
    """
    processor_names : List[str]
    processor_modalities : List[Modality]

class MergedProcessorOut:
    pass

class MergedProcessor:
    """
    Merged processor for multiple modalities

    :param processors: Each individual processor/tokenizer
    :param config: Config for merged processor
    """
    def __init__(self, processors, config : MergedProcessorConfig):
        # Bit janky but works with HF tokenizers/processors
        
        self.processors = processors
        self.processor_names = config.processor_names
        self.modalities = config.processor_modalities
    
    def __call__(
        self,
        images : Iterable[Image.Image] = None,
        text : Iterable[str] = None,
        audio : Any = None,
        video : Any = None,
        processor_kwargs : Dict[str, Dict] = {},
        return_tensors = "pt",
        return_dict = True,
        return_modality_dict = False
    ) -> MergedProcessorOut:
        """
        Call of the multimodal processor

        :param images: Images as list of PIL images
        :param text: List of strings
        :param audio: Audio in an iterable format specific to processor
        :param video: Video an an iterable format specific to processor
        :param processor_kwargs: Kwargs organized by processor names. i.e. is using SAMProcessor and GPT2Tokenizer, {'sam' : sam_kwargs, 'gpt2' : gpt2_kwargs}, kwargs will be passed to respective processors
        :param return_tensors: return tensors passed to each processor
        :param return_dict: Return dictionary with processor names as keys and their outputs as values. If false, just returns as a list.
        :param return_modality_dict: Alternate to above where we return dictionary with modalities as keys. 
        """

        if return_dict:
            output = {}
        else:
            output = []

        for (processor, name, modality) in zip(self.processors, self.processor_names, self.modalities):
            d = processor_kwargs[name] if name in processor_kwargs else {}
            if modality == Modality.IMAGE:
                inputs = images
            elif modality == Modality.TEXT:
                inputs = text
            elif modality == Modality.AUDIO:
                inputs = audio
            elif modality == Modality.VIDEO:
                inputs = video

            proc_out = processor(inputs, return_tensors = return_tensors, **d)

            if not return_dict:
                output.append(proc_out)
            elif return_modality_dict:
                output[modality] = proc_out
            else:
                output[name] = proc_out
            
        return output
                

        
        
        


