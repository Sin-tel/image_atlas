import hashlib
import time
from pathlib import Path
import numpy as np
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


WHITE_THRESH = 220
BLACK_THRESH = 20


def get_color(img):
    # (16, 16, 3)
    img = img.resize((16, 16), resample=0)
    arr = np.array(img)
    arr = arr[2:14, 2:14]

    # Mask out near-white and near-black pixels
    brightness = arr.mean(axis=2)
    mask = (brightness < WHITE_THRESH) & (brightness > BLACK_THRESH)

    if mask.sum() < 4:
        return arr.mean(axis=(0, 1))

    return arr[mask].mean(axis=0)


def get_image_metadata(path: Path) -> tuple[int, int, str]:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size

            avg = get_color(img)
            r, g, b = np.clip(avg, 0, 255).astype(int)

            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            return w, h, hex_color

    except Exception as e:
        log(f"{e}")
        return 100, 100, "#333333"
