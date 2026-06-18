"""PDF parser using PyMuPDF (fitz) with OCR fallback for scanned PDFs."""

import base64
import io
import uuid
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from src.core.parser import BaseParser
from src.core.schemas import RawDocument, DocumentImage
from src.registry import ModuleRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _vlm_extract_text(image_bytes: bytes, api_base: str, api_key: str) -> str:
    """Use VLM to extract text from a page image."""
    try:
        from openai import OpenAI
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        client = OpenAI(base_url=api_base, api_key=api_key)
        response = client.chat.completions.create(
            model="Qwen/Qwen3-VL-8B-Instruct",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请提取并输出这张图片中的所有文字内容，保持原文格式，不要添加任何解释。",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                ],
            }],
            max_tokens=2000,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"VLM text extraction failed: {e}")
        return f"[VLM 提取失败: {e}]"


@ModuleRegistry.parsers.register("pdf")
class PDFParser(BaseParser):
    """Parse PDF files using PyMuPDF. Falls back to VLM-OCR for scanned PDFs."""

    supported_extensions: List[str] = [".pdf"]

    def __init__(
        self,
        extract_images: bool = True,
        ocr_enabled: bool = True,
        vlm_api_base: str = "https://api.siliconflow.cn/v1",
        vlm_api_key: str = "",
    ):
        self.extract_images = extract_images
        self.ocr_enabled = ocr_enabled
        self.vlm_api_base = vlm_api_base
        self.vlm_api_key = vlm_api_key

    def parse(self, file_path: Path) -> RawDocument:
        logger.info(f"Parsing PDF: {file_path.name}")
        doc_id = str(uuid.uuid4())[:8]

        try:
            doc = fitz.open(str(file_path))
        except Exception as e:
            logger.error(f"Failed to open PDF: {file_path.name}: {e}")
            return RawDocument(
                doc_id=doc_id,
                source_path=str(file_path),
                content=f"[PDF 打开失败: {e}]",
                metadata={
                    "file_name": file_path.name,
                    "file_stem": file_path.stem,
                    "file_type": "pdf",
                    "page_count": 0,
                    "error": str(e),
                },
            )

        page_count = doc.page_count if hasattr(doc, 'page_count') else len(doc)
        all_text: List[str] = []
        images: List[DocumentImage] = []
        needs_ocr = False

        for page_num in range(page_count):
            try:
                page = doc.load_page(page_num)
                page_text = page.get_text("text")

                if page_text.strip():
                    all_text.append(f"[Page {page_num + 1}]\n{page_text.strip()}")
                elif self.ocr_enabled and self.vlm_api_key:
                    # Scanned page - render to image and use VLM to extract text
                    logger.info(f"Page {page_num+1} has no text layer, using VLM-OCR...")
                    needs_ocr = True
                    pix = page.get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")
                    ocr_text = _vlm_extract_text(
                        img_bytes, self.vlm_api_base, self.vlm_api_key
                    )
                    if ocr_text and not ocr_text.startswith("["):
                        all_text.append(f"[Page {page_num + 1} (OCR)]\n{ocr_text.strip()}")
                    else:
                        all_text.append(f"[Page {page_num + 1} - OCR 未识别到文字]")
                else:
                    all_text.append(f"[Page {page_num + 1} - 无文字层，{'' if self.ocr_enabled else 'OCR 未启用'}]")

                # Extract embedded images
                if self.extract_images:
                    for img_index, img in enumerate(page.get_images(full=True)):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            doc_image = DocumentImage(
                                image_id=f"{doc_id}_p{page_num+1}_img{img_index}",
                                image_bytes=base_image["image"],
                                page_num=page_num + 1,
                            )
                            images.append(doc_image)
                        except Exception as e:
                            logger.warning(f"Image extract failed p{page_num+1}: {e}")
            except Exception as e:
                logger.warning(f"Failed to process page {page_num+1}: {e}")
                all_text.append(f"[Page {page_num + 1} - 解析失败: {e}]")

        doc.close()
        full_text = "\n\n".join(all_text)

        metadata = {
            "file_name": file_path.name,
            "file_stem": file_path.stem,
            "file_type": "pdf",
            "page_count": page_count,
            "ocr_used": needs_ocr,
        }

        logger.info(
            f"PDF parsed: {len(full_text)} chars, {page_count} pages"
            f"{' (OCR used)' if needs_ocr else ''}"
        )
        return RawDocument(
            doc_id=doc_id,
            source_path=str(file_path),
            content=full_text,
            metadata=metadata,
            images=images,
        )
