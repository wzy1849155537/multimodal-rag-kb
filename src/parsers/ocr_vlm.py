"""OCR + VLM pipeline for image understanding."""

from typing import Optional

from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)


class OCRVLMPipeline:
    """Pipeline: OCR text extraction + VLM semantic description for images."""

    def __init__(
        self,
        ocr_lang: str = "ch",
        use_gpu: bool = False,
        vlm_api_base: Optional[str] = None,
        vlm_api_key: Optional[str] = None,
        vlm_model: str = "qwen-vl-plus",
    ):
        self.ocr_lang = ocr_lang
        self.use_gpu = use_gpu
        self.vlm_api_base = vlm_api_base
        self.vlm_api_key = vlm_api_key
        self.vlm_model = vlm_model

        self._ocr = None
        self._ocr_initialized = False

    def _init_ocr(self):
        """Lazy-init PaddleOCR."""
        if self._ocr_initialized:
            return
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                lang=self.ocr_lang,
                use_angle_cls=True,
                use_gpu=self.use_gpu,
                show_log=False,
            )
            self._ocr_initialized = True
            logger.info("PaddleOCR initialized")
        except ImportError:
            logger.warning(
                "PaddleOCR not installed. Install with: pip install paddleocr"
            )
            self._ocr = None
            self._ocr_initialized = True

    def extract_text(self, image: Image.Image) -> str:
        """Extract text from an image using OCR."""
        self._init_ocr()
        if self._ocr is None:
            return "[OCR 未安装 — 请安装 paddleocr]"

        import numpy as np
        img_array = np.array(image)
        try:
            result = self._ocr.ocr(img_array, cls=True)
            if not result or not result[0]:
                return ""
            texts = []
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]
                    confidence = line[1][1]
                    if confidence > 0.5:
                        texts.append(text)
            return "\n".join(texts)
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""

    def describe_image(self, image: Image.Image) -> str:
        """Generate a semantic description of the image using VLM."""
        if not self.vlm_api_key:
            return "[VLM 未配置 — 请设置 API Key]"

        import base64
        import io
        from openai import OpenAI

        # Convert PIL image to base64
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        try:
            client = OpenAI(
                base_url=self.vlm_api_base,
                api_key=self.vlm_api_key,
            )
            response = client.chat.completions.create(
                model=self.vlm_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "请详细描述这张图片的内容。"
                                "如果是图表，请描述图表类型、关键数据点和趋势。"
                                "如果是文档截图，请描述文档的结构和主要内容。"
                                "如果是照片，请描述场景、物体和关键特征。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            },
                        },
                    ],
                }],
                max_tokens=500,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"VLM description failed: {e}")
            return f"[VLM 错误: {e}]"
