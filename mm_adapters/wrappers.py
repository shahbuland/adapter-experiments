from transformers import (
    SamImageProcessor,
    WhisperModel
)

import torch

class SamVisionProcessor(SamImageProcessor):
    def __call__(self, *args, **kwargs):
        res = super().__call__(*args, **kwargs)
        del res['original_sizes'], res['reshaped_input_sizes']
        return res

class WhisperWrapper(WhisperModel):
    def forward(self, input_featues, *args, **kwargs):
        device = input_features.device
        dtype = input_features.dtype
        d_ids = torch.tensor([[1,1]]) * self.config.decoder_start_token_id
        kwargs['decoder_input_ids'] = d_ids.to(device).to(dtype)

        super().forward(input_featues, *args, **kwargs)