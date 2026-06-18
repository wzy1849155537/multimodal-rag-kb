"""Create a desktop shortcut for the RAG Knowledge Base app."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
DESKTOP = Path(os.environ.get("USERPROFILE", "")) / "Desktop"
APP_NAME = "RAG知识库问答系统"


def create_icon():
    """Generate a simple .ico file programmatically."""
    from PIL import Image, ImageDraw, ImageFont

    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background circle - gradient blue
    draw.ellipse([10, 10, size - 10, size - 10], fill="#1E40AF")
    draw.ellipse([30, 30, size - 30, size - 30], fill="#3B82F6")
    draw.ellipse([50, 50, size - 50, size - 50], fill="#60A5FA")

    # "RAG" text
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 80)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), "RAG", font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) // 2
    y = (size - th) // 2 - 10
    draw.text((x, y), "RAG", fill="white", font=font)

    # "KB" subtext
    try:
        font_small = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 36)
    except Exception:
        font_small = font
    bbox2 = draw.textbbox((0, 0), "KB", font=font_small)
    tw2, th2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
    draw.text(
        ((size - tw2) // 2, y + th + 5),
        "KB", fill="white", font=font_small
    )

    ico_path = PROJECT_ROOT / "app_icon.ico"
    # Save as ICO - use multiple sizes
    img.save(ico_path, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    print(f"[OK] Icon created: {ico_path}")
    return ico_path


def create_shortcut(ico_path: Path):
    """Create a desktop shortcut."""
    import pythoncom
    from win32com.client import Dispatch

    shortcut_path = DESKTOP / f"{APP_NAME}.lnk"

    # Target: pythonw.exe (no console window)
    pythonw = Path(sys.executable).parent / "pythonw.exe"
    target = str(pythonw)
    args = f'"{PROJECT_ROOT / "desktop_app.py"}"'
    work_dir = str(PROJECT_ROOT)

    shell = Dispatch("WScript.Shell")
    shortcut = shell.CreateShortcut(str(shortcut_path))
    shortcut.TargetPath = target
    shortcut.Arguments = args
    shortcut.WorkingDirectory = work_dir
    shortcut.IconLocation = str(ico_path)
    shortcut.Description = "多模态RAG知识库问答系统"
    shortcut.Save()

    print(f"[OK] Shortcut created: {shortcut_path}")


def main():
    print(f"Creating {APP_NAME} desktop shortcut...")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Desktop: {DESKTOP}")
    print()

    ico_path = create_icon()
    create_shortcut(ico_path)

    print()
    print("=" * 40)
    print("  Done! Double-click the icon on your desktop:")
    print(f"  {APP_NAME}")
    print("=" * 40)


if __name__ == "__main__":
    main()
