from typing import Dict, Union, Any, List

import torch
from enum import Enum

class Modality(Enum):
    IMAGE = "image"
    AUDIO = "audio"
    TEXT = "text"
    VIDEO = "video"

def dict_to(d : Dict, dest : Union[Any, List[Any]]):
    """
    Allows us to change device/datatype over an arbitrary dictionary of tensors. dest is argument of tensor.to(...)
    Allows for nested types (i.e. if a dict value is itself a dict or a list)
    Skips specific tensor/device/dtype combinations that don't make sense
    """
    
    if not isinstance(dest, list):
        dest = [dest]

    def is_valid_match(t, dest_j):
        if (t.dtype == torch.long or t.dtype == torch.bool) and dest_j == torch.half:
            return False
        return True
    
    # To account for when theres lists of lists or nested data types in the dict
    def recursive_cast(x, dest : List):
        if torch.is_tensor(x):
            for dest_i in dest:
                if is_valid_match(x, dest_i):
                    x = x.to(dest_i)
            return x
        elif isinstance(x, list):
            return [recursive_cast(x_i, dest) for x_i in x]
        elif isinstance(x, dict):
            return {k: recursive_cast(v, dest) for k, v in x.items()}
        else:
            return x

    return recursive_cast(d, dest)

def freeze_module(module : torch.nn.Module):
    for p in module.parameters():
        p.require_grad = False

def unfreeze_module(module : torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def remove_suffixes(s):
    """
    Remove common suffixes from huggingface names
    """
    s = s.lower()
    s = s.replace("processor", "")
    s = s.replace("tokenizer", "")
    s = s.replace("language", "")
    s = s.replace("vision", "")
    return s