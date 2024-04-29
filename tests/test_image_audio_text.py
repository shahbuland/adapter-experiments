"""
Test an adapter that merges SIGLIP/DETR feature maps and also Whisper/CLAP for audio
"""

from mm_adapters import (
    MergedAdapter,
    MergedAdapterConfig,
    MergedProcessor,
    MergedProcessorConfig,
    Modality
)

from mm_adapters.wrappers import SamVisionProcessor, WhisperWrapper
import torchaudio
import numpy as np

if __name__ == "__main__":
    from transformers import (
        SamModel, AutoModel, AutoProcessor, AutoTokenizer,
        SiglipVisionModel, SiglipImageProcessor,
        WhisperProcessor, WhisperModel,
        ClapAudioModel, ClapFeatureExtractor
    )

    # All the ids first
    detr_id = "facebook/detr-resnet-50"
    siglip_id = "google/siglip-base-patch16-224"
    slm_id = "stabilityai/stablelm-2-1_6b" 
    whisper_id = "openai/whisper-tiny"
    clap_id = "laion/larger_clap_music"

    # Then all the processors
    detr_proc = AutoProcessor.from_pretrained(detr_id)
    siglip_proc = SiglipImageProcessor.from_pretrained(siglip_id)
    slm_proc = AutoTokenizer.from_pretrained(slm_id)
    slm_proc.pad_token_id = slm_proc.eos_token_id
    whisper_proc = WhisperProcessor.from_pretrained(whisper_id)
    clap_proc = ClapFeatureExtractor.from_pretrained(clap_id)

    # Then all the models
    detr_model = AutoModel.from_pretrained(detr_id)
    siglip_model = SiglipVisionModel.from_pretrained(siglip_id)
    whisper_model = WhisperModel.from_pretrained(whisper_id).encoder
    clap_model = ClapAudioModel.from_pretrained(clap_id)

    # Processing
    names = ['detr', 'siglip', 'whisper', 'clap', 'slm']
    proc_config = MergedProcessorConfig(
        names,
        [Modality.IMAGE]*2 + [Modality.AUDIO]*2 + [Modality.TEXT]
    )
    processor = MergedProcessor(
        [detr_proc, siglip_proc, whisper_proc, clap_proc, slm_proc],
        proc_config
    )

    from PIL import Image

    images = [Image.open("sample.png")]*2
    text = ["hello world"] * 2
    audio, sr = torchaudio.load("sample.wav", normalize = True)
    audio = audio[:,:4*sr] # only 4 seconds for simplicity
    audio = audio.mean(0).numpy() # mono
    audio = np.stack([audio]*2)
    proc_out = processor(
        images, text, audio,
        processor_kwargs = {
            'slm' : {'padding' : 'max_length', 'max_length' : 64, 'truncation' : True},
            'whisper' : {'sampling_rate' : 16000},
            'clap' : {'sampling_rate' : 48000}  
        },
        return_tensors = "pt"
    )

    print(proc_out.keys())

    # Adapter
    adapter_config = MergedAdapterConfig(
        names[:-1],
        use_abstractor = True,
        use_type_embeddings = True
    )

    adapter = MergedAdapter(
        [detr_model, siglip_model, whisper_model, clap_model],
        adapter_config
    )

    adapter_out = adapter(
        proc_out
    )

    print(adapter_out.shape)
    print("Success!")
    