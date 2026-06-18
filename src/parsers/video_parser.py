"""Video parser: extract audio, transcribe with ASR, index transcript."""

import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List

from src.core.parser import BaseParser
from src.core.schemas import RawDocument
from src.registry import ModuleRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _transcribe_audio(audio_path: Path, api_base: str, api_key: str) -> str:
    """Transcribe audio using SiliconFlow SenseVoice API."""
    from openai import OpenAI

    client = OpenAI(base_url=api_base, api_key=api_key)

    with open(audio_path, "rb") as f:
        try:
            response = client.audio.transcriptions.create(
                model="FunAudioLLM/SenseVoiceSmall",
                file=f,
                response_format="text",
            )
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return f"[语音转写失败: {e}]"


@ModuleRegistry.parsers.register("video")
class VideoParser(BaseParser):
    """Parse video files by extracting audio and transcribing."""

    supported_extensions: List[str] = [
        ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"
    ]

    def __init__(
        self,
        asr_api_base: str = "https://api.siliconflow.cn/v1",
        asr_api_key: str = "",
    ):
        self.asr_api_base = asr_api_base
        self.asr_api_key = asr_api_key

    def _find_ffmpeg(self) -> str:
        """Find ffmpeg binary."""
        # Hardcoded known-good path from winget install
        ff = Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.1-full_build/bin/ffmpeg.exe"
        if ff.exists():
            return str(ff)
        return "ffmpeg"

    def parse(self, file_path: Path) -> RawDocument:
        logger.info(f"Parsing video: {file_path.name}")
        doc_id = str(uuid.uuid4())[:8]

        duration = ""
        transcript = ""

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / f"{doc_id}_audio.mp3"

            ffmpeg_bin = self._find_ffmpeg()
            # Replace only the filename, not the path
            ff_dir = str(Path(ffmpeg_bin).parent)
            ffprobe_bin = str(Path(ff_dir) / "ffprobe.exe")

            # Extract audio with ffmpeg
            try:
                subprocess.run(
                    [
                        ffmpeg_bin, "-i", str(file_path),
                        "-vn", "-acodec", "libmp3lame",
                        "-q:a", "5", "-y",
                        str(audio_path),
                    ],
                    capture_output=True,
                    timeout=300,  # 5 min timeout for large videos
                    check=True,
                )

                # Get duration
                result = subprocess.run(
                    [
                        ffprobe_bin, "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        str(file_path),
                    ],
                    capture_output=True, text=True,
                )
                seconds = float(result.stdout.strip())
                mins, secs = divmod(int(seconds), 60)
                hours, mins = divmod(mins, 60)
                if hours:
                    duration = f"{hours}小时{mins}分{secs}秒"
                else:
                    duration = f"{mins}分{secs}秒"

                # Transcribe
                if self.asr_api_key and audio_path.stat().st_size > 0:
                    logger.info(f"Transcribing {duration} audio...")
                    transcript = _transcribe_audio(
                        audio_path, self.asr_api_base, self.asr_api_key
                    )
                    logger.info(f"Transcription: {len(transcript)} chars")
                else:
                    transcript = "[语音转写未配置 API Key]"

            except subprocess.CalledProcessError as e:
                logger.error(f"ffmpeg failed: {e.stderr.decode() if e.stderr else e}")
                transcript = "[音频提取失败：请安装 ffmpeg]"
            except FileNotFoundError:
                transcript = "[需要安装 ffmpeg：https://ffmpeg.org/download.html]"
            except Exception as e:
                logger.error(f"Video processing failed: {e}")
                transcript = f"[视频处理失败: {e}]"

        content = (
            f"[视频文件: {file_path.name}]\n"
            f"[时长: {duration}]\n\n"
            f"[语音转写内容]\n{transcript}"
        )

        metadata = {
            "file_name": file_path.name,
            "file_stem": file_path.stem,
            "file_type": "video",
            "duration": duration,
            "transcript_length": len(transcript),
        }

        logger.info(
            f"Video parsed: {file_path.name} ({duration}), "
            f"transcript: {len(transcript)} chars"
        )
        return RawDocument(
            doc_id=doc_id,
            source_path=str(file_path),
            content=content,
            metadata=metadata,
        )
