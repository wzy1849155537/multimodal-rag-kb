"""Standalone image parser using OCR + VLM."""

import uuid
from pathlib import Path
from typing import List

from PIL import Image

from src.core.parser import BaseParser
from src.core.schemas import RawDocument, DocumentImage
from src.registry import ModuleRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


@ModuleRegistry.parsers.register("image")
class ImageParser(BaseParser):
    """Parse standalone image files with OCR + VLM."""

    supported_extensions: List[str] = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

    def __init__(
        self,
        ocr_lang: str = "ch",
        use_gpu: bool = False,
        vlm_api_base: str = "",
        vlm_api_key: str = "",
        vlm_model: str = "qwen-vl-plus",
    ):
        self.ocr_lang = ocr_lang
        self.use_gpu = use_gpu
        self.vlm_api_base = vlm_api_base
        self.vlm_api_key = vlm_api_key
        self.vlm_model = vlm_model

    def parse(self, file_path: Path) -> RawDocument:
        logger.info(f"Parsing image: {file_path.name}")
        doc_id = str(uuid.uuid4())[:8]

        image = Image.open(file_path)
        doc_image = DocumentImage(
            image_id=f"{doc_id}_img0",
            image_path=file_path,
            page_num=1,
        )

        # OCR + VLM (lazy, only if configured)
        from src.parsers.ocr_vlm import OCRVLMPipeline
        pipeline = OCRVLMPipeline(
            ocr_lang=self.ocr_lang,
            use_gpu=self.use_gpu,
            vlm_api_base=self.vlm_api_base,
            vlm_api_key=self.vlm_api_key,
            vlm_model=self.vlm_model,
        )

        ocr_text = pipeline.extract_text(image)
        doc_image.ocr_text = ocr_text

        vlm_desc = pipeline.describe_image(image)
        doc_image.vlm_description = vlm_desc

        # Build content from OCR + VLM
        content_parts = []
        if ocr_text:
            content_parts.append(f"[OCR 文字提取]\n{ocr_text}")
        if vlm_desc:
            content_parts.append(f"[VLM 图像描述]\n{vlm_desc}")

        content = "\n\n".join(content_parts) if content_parts else "[无文字内容]"

        metadata = {
            "file_name": file_path.name,
            "file_stem": file_path.stem,
            "file_type": "image",
            "image_width": image.width,
            "image_height": image.height,
        }

        logger.info(
            f"Image parsed: {len(ocr_text)} OCR chars, "
            f"{len(vlm_desc)} VLM chars from {file_path.name}"
        )
        return RawDocument(
            doc_id=doc_id,
            source_path=str(file_path),
            content=content,
            metadata=metadata,
            images=[doc_image],
        )
