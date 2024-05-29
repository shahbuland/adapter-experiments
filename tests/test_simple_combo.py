"""
Test an adapter that merges SIGLIP/SAM/DETR feature maps
"""

from mm_adapters import (
    MergedAdapter,
    MergedAdapterConfig,
    MergedProcessor,
    MergedProcessorConfig,
    Modality
)

from mm_adapters.wrappers import SamVisionProcessor

if __name__ == "__main__":
    from transformers import (
        AutoModel, AutoProcessor, AutoTokenizer,
        SamModel, SiglipVisionModel, SiglipImageProcessor,
        AutoImageProcessor,
    )

    sam_id = "nielsr/slimsam-50-uniform"
    siglip_id = "google/siglip-base-patch16-224"
    slm_id = "stabilityai/stablelm-2-1_6b" 

    sam_model = SamModel.from_pretrained(sam_id).vision_encoder
    siglip_model = SiglipVisionModel.from_pretrained(siglip_id)

    sam_proc = SamVisionProcessor.from_pretrained(sam_id)
    siglip_proc = SiglipImageProcessor.from_pretrained(siglip_id)
    slm_proc = AutoTokenizer.from_pretrained(slm_id)
    slm_proc.pad_token_id = slm_proc.eos_token_id

    names = ["siglip", "sam","slm"]
    proc_config = MergedProcessorConfig(
        names,
        [Modality.IMAGE]*2 + [Modality.TEXT]
    )
    processor = MergedProcessor(
        [siglip_proc, sam_proc, slm_proc],
        proc_config
    )
    # Processing text
    from PIL import Image

    images = [Image.open("sample.png")]*2
    text = ["hello world"] * 2

    proc_out = processor(
        images, text,
        processor_kwargs = {'slm' : {'padding' : 'max_length', 'max_length' : 64, 'truncation' : True}},
        return_tensors = "pt"
    )
    # Adapter stuff

    adapter_config = MergedAdapterConfig(
        names[:-1],
        use_abstractor = True,
        use_type_embeddings = True
    )
    adapter = MergedAdapter(
        [siglip_model, sam_model],
        adapter_config
    )

    adapter_out = adapter(
        proc_out
    )

    print(adapter_out.shape)
