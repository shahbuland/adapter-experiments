"""
Test an adapter that merges SIGLIP/SAM/DETR feature maps
"""

from mm_adapters import (
    MergedAdapter,
    MergedAdapterConfig,
    MergedProcessor,
    MergredProcessorConfig,
    Modality
)

if __name__ == "__main__":
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    from transformers import SiglipVisionModel, SiglipImageProcessor

    detr_id = "facebook/detr-resnet-50"
    sam_id = "facebook/sam-vit-huge"
    siglip_id = "google/siglip-base-patch16-224"
    slm_id = "stabilityai/stablelm-2-1_6b" 

    detr_model = AutoModel.from_pretrained(detr_id)
    sam_model = AutoModel.from_pretrained(sam_id)
    siglip_model = SiglipVisionModel.from_pretrained(siglip_id)

    detr_proc = AutoProcessor.from_pretrained(detr_id)
    sam_proc = AutoProcessor.from_pretrained(sam_id)
    siglip_proc = SiglipImageProcessor.from_pretrained(siglip_id)
    slm_proc = AutoTokenizer.from_pretrained(slm_id)

    names = ["detr", "sam", "siglip", "slm"]
    proc_config = MergedProcessorConfig(
        ["detr", "sam", "siglip", "slm"],
        [Modality.IMAGES]*3 + [Modality.TEXT]
    )
    processor = MergedProcessor(
        [detr_proc, sam_proc, siglip_proc, slm_proc],
        proc_config
    )

    # Processing text
    from PIL import Image

    images = [Image.open("sample.png")]*2
    text = ["hello world"] * 2

    proc_out = processor(
        images, text,
        processor_kwargs = {'slm' : {'padding' : 'max_length', 'max_length' : 64, 'truncation' : True}}
    )
    
    print(proc_out)

    # Adapter stuff

    adapter_config = MergedAdapterConfig(
        names[:-1],
        use_abstractor = True,
        use_type_embeddings = True
    )
    adapter = MergedAdapter(
        [detr_model, sam_model, siglip_model],
        adapter_config
    )

    adapter_out = adapter(
        proc_out
    )

    print(adapter_out.shape)
    # Processing =



