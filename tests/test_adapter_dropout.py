"""
Test the ability to drop out adapters
In some cases where there is image data
where we are certain it has no text,
it might make sense to skip OCR resultss
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
        Dinov2Model, AutoImageProcessor,
        VisionEncoderDecoderModel,
        NougatProcessor
    )

    siglip_id = "google/siglip-base-patch16-224"
    ocr_id = "facebook/nougat-base"
    slm_id = "stabilityai/stablelm-2-1_6b"
    
    siglip_model = SiglipVisionModel.from_pretrained(siglip_id)
    ocr_model = VisionEncoderDecoderModel.from_pretrained(ocr_id).encoder

    siglip_proc = SiglipImageProcessor.from_pretrained(siglip_id)
    slm_proc = AutoTokenizer.from_pretrained(slm_id)
    slm_proc.pad_token_id = slm_proc.eos_token_id
    ocr_proc = NougatProcessor.from_pretrained(ocr_id)

    names = ["siglip", "nougat", "slm"]
    proc_config = MergedProcessorConfig(
        names,
        [Modality.IMAGE]*2 + [Modality.TEXT]
    )
    processor = MergedProcessor(
        [siglip_proc, ocr_proc, slm_proc],
        proc_config
    )

    # Processing text
    from PIL import Image

    # Simulate batch of size 2
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
        [siglip_model, ocr_model],
        adapter_config
    )

    adapter_out = adapter(
        proc_out
    )
    print(adapter_out.shape)

    ocr_drop = adapter(
        proc_out, ["nougat"] # Drop nougat
    )

    print(ocr_drop.shape)