import hashlib
import time
from pathlib import Path
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def file_cache_key(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}|{stat.st_mtime}|{stat.st_size}"
    return hashlib.sha1(raw.encode()).hexdigest()


def find_images(paths: list[Path], recursive: bool) -> list[Path]:
    result: list[Path] = []
    for root in paths:
        pattern = "**/*" if recursive else "*"
        result.extend(
            p
            for p in root.glob(pattern)
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    return sorted(set(result))


def get_image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return 100, 100
