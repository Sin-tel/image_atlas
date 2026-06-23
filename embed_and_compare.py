"""
imagepile POC — embedding quality test
========================================

Scans a folder of images, embeds every image with two models
(DINOv2 and SigLIP2), caches the embeddings to disk, and
generates a static HTML page where you can click any image and see
its top-K nearest neighbors under each model, side by side.

The point of this script is ONLY to answer: "which embedding model
gives nearest-neighbor results that actually look right on MY images?"
Nothing here is meant to be the final app.

Usage:
    python embed_and_compare.py --folder "D:/Pictures/some_folder" --recursive

First run will download model weights (a few GB total) and embed
every image — this is the slow part. Subsequent runs on the same
folder reuse the cache and only embed new/changed files.

Requirements (install once):
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers pillow numpy tqdm

    (use whatever cu1xx tag matches your installed CUDA; check
    https://pytorch.org/get-started/locally/ if unsure)
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from config import access_token

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}

DINOV2_MODEL_ID = "facebook/dinov2-small"
# SIGLIP_MODEL_ID = "google/siglip2-large-patch16-256"
SIGLIP_MODEL_ID = "google/siglip2-base-patch16-256"

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# File discovery + cache keying
# ---------------------------------------------------------------------------

def find_images(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    paths = [
        p for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths)


def file_cache_key(path: Path) -> str:
    """Key on path + mtime + size so edited/replaced files re-embed,
    unchanged files reuse cache, and renamed-but-identical files don't
    falsely share a cache entry (good enough for POC; the real app can
    use content hashing for true dedup)."""
    stat = path.stat()
    raw = f"{path.resolve()}|{stat.st_mtime}|{stat.st_size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Embedding cache (one JSON-lines manifest + one .npy blob per model)
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """Stores embeddings for one model. Keyed by file_cache_key.
    Persisted as a single .npz so it's trivial to load/save."""

    def __init__(self, name: str, dim: int):
        self.name = name
        self.dim = dim
        self.path = CACHE_DIR / f"{name}.npz"
        self.keys: dict[str, int] = {}      # cache_key -> row index
        self.paths: dict[str, str] = {}      # cache_key -> original path (for display)
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        data = np.load(self.path, allow_pickle=True)
        self.vectors = data["vectors"]
        keys = data["keys"].tolist()
        paths = data["paths"].tolist()
        self.keys = {k: i for i, k in enumerate(keys)}
        self.paths = dict(zip(keys, paths))
        log(f"[{self.name}] loaded {len(self.keys)} cached embeddings")

    def save(self):
        keys = list(self.keys.keys())
        ordered = np.zeros((len(keys), self.dim), dtype=np.float32)
        paths = []
        for k in keys:
            ordered[self.keys[k]] = self.vectors[self.keys[k]]
            paths.append(self.paths[k])
        np.savez_compressed(
            self.path,
            vectors=ordered,
            keys=np.array(keys, dtype=object),
            paths=np.array(paths, dtype=object),
        )

    def has(self, key: str) -> bool:
        return key in self.keys

    def add(self, key: str, path: str, vector: np.ndarray):
        idx = len(self.keys)
        self.keys[key] = idx
        self.paths[key] = path
        if idx >= self.vectors.shape[0]:
            # grow buffer
            grown = np.zeros((max(idx * 2, 64), self.dim), dtype=np.float32)
            grown[: self.vectors.shape[0]] = self.vectors
            self.vectors = grown
        self.vectors[idx] = vector

    def matrix(self):
        """Return (paths_list, NxD normalized matrix) trimmed to actual size."""
        n = len(self.keys)
        paths_list = [None] * n
        for k, i in self.keys.items():
            paths_list[i] = self.paths[k]
        mat = self.vectors[:n]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return paths_list, mat / norms


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_dino(device):
    from transformers import AutoImageProcessor, AutoModel
    log(f"Loading DINOv2 ({DINOV2_MODEL_ID}) ...")
    processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID, token=access_token)
    model = AutoModel.from_pretrained(DINOV2_MODEL_ID, token=access_token).to(device).eval()
    return processor, model


def load_siglip(device):
    from transformers import AutoProcessor, AutoModel
    log(f"Loading SigLIP2 ({SIGLIP_MODEL_ID}) ...")
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID, token=access_token)
    model = AutoModel.from_pretrained(SIGLIP_MODEL_ID, token=access_token).to(device).eval()
    return processor, model


def embed_dino(processor, model, device, images):
    import torch
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inputs)
        feats = out.last_hidden_state[:, 0, :]
        # mean-pooling patch tokens
        # feats = out.last_hidden_state[:, 1:, :].mean(dim=1)
    return feats.float().cpu().numpy()


def embed_siglip(processor, model, device, images):
    import torch
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.get_image_features(**inputs)

    feats = out.pooler_output
    return feats.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Main embedding pass
# ---------------------------------------------------------------------------

def run_embedding(model_name, load_fn, embed_fn, image_paths, device, batch_size, dim):
    from PIL import Image

    cache = EmbeddingCache(model_name, dim)
    todo = [p for p in image_paths if not cache.has(file_cache_key(p))]

    if not todo:
        log(f"[{model_name}] nothing new to embed ({len(image_paths)} total, all cached)")
        return cache

    log(f"[{model_name}] embedding {len(todo)} new images ({len(image_paths) - len(todo)} already cached)")
    processor, model = load_fn(device)

    from tqdm import tqdm

    for i in tqdm(range(0, len(todo), batch_size), desc=model_name):
        batch_paths = todo[i : i + batch_size]
        batch_imgs = []
        valid_paths = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_imgs.append(img)
                valid_paths.append(p)
            except Exception as e:
                log(f"  skip (unreadable): {p} ({e})")
        if not batch_imgs:
            continue

        try:
            vectors = embed_fn(processor, model, device, batch_imgs)
        except Exception as e:
            log(f"  batch failed, falling back to one-by-one: {e}")
            vectors = []
            for img in batch_imgs:
                try:
                    vectors.append(embed_fn(processor, model, device, [img])[0])
                except Exception as e2:
                    log(f"    failed on single image too, skipping: {e2}")
                    vectors.append(None)

        for p, v in zip(valid_paths, vectors):
            if v is None:
                continue
            cache.add(file_cache_key(p), str(p), np.asarray(v, dtype=np.float32))

    cache.save()
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return cache


# ---------------------------------------------------------------------------
# Nearest neighbors
# ---------------------------------------------------------------------------

def top_k_neighbors(matrix: np.ndarray, k: int):
    """matrix: NxD, L2-normalized. Returns NxK indices of nearest neighbors
    (excluding self), via brute-force cosine similarity."""
    sims = matrix @ matrix.T
    np.fill_diagonal(sims, -np.inf)
    idx = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
    # sort the top-k by actual similarity, descending
    row_idx = np.arange(matrix.shape[0])[:, None]
    order = np.argsort(-sims[row_idx, idx], axis=1)

    # print(sims.shape)
    idx = idx[row_idx, order]
    # print(sims[row_idx, idx])
    return idx


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>imagepile POC — embedding comparison</title>
<style>
  body {{ background: #111; color: #ddd; font-family: system-ui, sans-serif; margin: 0; padding: 16px; }}
  h1 {{ font-size: 16px; color: #888; font-weight: normal; }}
  #grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 4px; }}
  #grid img {{ width: 100%; height: 120px; object-fit: cover; cursor: pointer; border-radius: 4px; opacity: 0.9; }}
  #grid img:hover {{ opacity: 1; outline: 2px solid #5af; }}
  #detail {{ position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.95);
             display: none; flex-direction: column; padding: 20px; overflow-y: auto; z-index: 10; }}
  #detail.open {{ display: flex; }}
  #detail .close {{ position: absolute; top: 12px; right: 20px; font-size: 28px; cursor: pointer; color: #aaa; }}
  #detail .query-row {{ display: flex; gap: 16px; align-items: flex-start; margin-bottom: 24px; }}
  #detail .query-row img {{ max-height: 280px; max-width: 360px; border-radius: 6px; }}
  .columns {{ display: flex; gap: 24px; flex-wrap: wrap; }}
  .column {{ flex: 1; min-width: 320px; }}
  .column h3 {{ color: #5af; font-weight: normal; font-size: 14px; margin-bottom: 8px; }}
  .nbrow {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px; }}
  .nbrow img {{ width: 100%; height: 90px; object-fit: cover; border-radius: 4px; cursor: pointer; }}
  .path {{ font-size: 10px; color: #666; word-break: break-all; max-width: 200px; }}
  .info {{ color: #888; font-size: 12px; margin-bottom: 12px; }}
</style>
</head>
<body>
<h1>{count} images — click any thumbnail to compare nearest neighbors across models</h1>
<div id="grid"></div>

<div id="detail">
  <span class="close" onclick="closeDetail()">&times;</span>
  <div class="query-row">
    <img id="query-img" src="">
    <div>
      <div style="color:#5af; font-size:14px;">Query image</div>
      <div class="path" id="query-path"></div>
    </div>
  </div>
  <div class="columns" id="columns"></div>
</div>

<script>
const DATA = {data_json};
const grid = document.getElementById('grid');
const detail = document.getElementById('detail');
const columnsEl = document.getElementById('columns');

DATA.images.forEach((img, i) => {{
  const el = document.createElement('img');
  el.src = img.url;
  el.loading = 'lazy';
  el.onclick = () => openDetail(i);
  grid.appendChild(el);
}});

function openDetail(i) {{
  const img = DATA.images[i];
  document.getElementById('query-img').src = img.url;
  document.getElementById('query-path').textContent = img.path;
  columnsEl.innerHTML = '';

  for (const modelName of DATA.models) {{
    const col = document.createElement('div');
    col.className = 'column';
    const h3 = document.createElement('h3');
    h3.textContent = modelName;
    col.appendChild(h3);

    const row = document.createElement('div');
    row.className = 'nbrow';
    const neighbors = img.neighbors[modelName] || [];
    neighbors.forEach(nIdx => {{
      const nImg = DATA.images[nIdx];
      const nEl = document.createElement('img');
      nEl.src = nImg.url;
      nEl.title = nImg.path;
      nEl.onclick = () => openDetail(nIdx);
      row.appendChild(nEl);
    }});
    col.appendChild(row);
    columnsEl.appendChild(col);
  }}

  detail.classList.add('open');
}}

function closeDetail() {{
  detail.classList.remove('open');
}}
</script>
</body>
</html>
"""


def file_url(path: Path) -> str:
    # file:// URL for local browser viewing
    return path.resolve().as_uri()


def build_html(output_path: Path, caches: dict, all_paths: list[Path], k: int):
    """caches: {model_name: EmbeddingCache}. Builds neighbor lists per model
    for the intersection of images present in ALL caches (so comparison is fair),
    using a shared index order."""

    # Use the path itself (not cache key) as the join key across models
    common_paths = None
    per_model_lookup = {}
    for name, cache in caches.items():
        paths_list, mat = cache.matrix()
        lookup = {p: i for i, p in enumerate(paths_list)}
        per_model_lookup[name] = (paths_list, mat, lookup)
        s = set(paths_list)
        common_paths = s if common_paths is None else (common_paths & s)

    common_paths = sorted(common_paths)
    log(f"Building HTML for {len(common_paths)} images present in all model caches")

    if not common_paths:
        log("ERROR: no images are present in all caches — did embedding fail for one model?")
        return

    # global index for the viewer (shared across models)
    global_index = {p: i for i, p in enumerate(common_paths)}

    images_json = [
        {"url": file_url(Path(p)), "path": p, "neighbors": {}}
        for p in common_paths
    ]

    for name, (paths_list, mat, lookup) in per_model_lookup.items():
        # subset + reorder matrix to common_paths order
        rows = [lookup[p] for p in common_paths]
        sub = mat[rows]
        nbrs = top_k_neighbors(sub, k)
        for i, nb_row in enumerate(nbrs):
            images_json[i]["neighbors"][name] = [int(x) for x in nb_row]

    data = {"models": list(caches.keys()), "images": images_json}
    html = HTML_TEMPLATE.format(count=len(common_paths), data_json=json.dumps(data))
    output_path.write_text(html, encoding="utf-8")
    log(f"Wrote {output_path.resolve()}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, type=str)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="cap number of images for a quick test run (0 = no cap)")
    ap.add_argument("--output", type=str, default="comparison.html")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        log(f"ERROR: folder does not exist: {folder}")
        sys.exit(1)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")
    if device == "cpu":
        log("WARNING: no CUDA GPU detected — this will be slow. Check your torch install matches your CUDA version.")

    log(f"Scanning {folder} (recursive={args.recursive}) ...")
    image_paths = find_images(folder, args.recursive)
    log(f"Found {len(image_paths)} images")

    if args.limit:
        image_paths = image_paths[: args.limit]
        log(f"Limiting to first {len(image_paths)} for this run")

    if not image_paths:
        log("No images found, exiting.")
        return

    dinov2_cache = run_embedding(
        "dinov2_large", load_dino, embed_dino,
        image_paths, device, args.batch_size, dim=384,
        # image_paths, device, args.batch_size, dim=1024,
    )
    siglip_cache = run_embedding(
        "siglip2_base", load_siglip, embed_siglip,
        image_paths, device, args.batch_size, dim=768,
        # image_paths, device, args.batch_size, dim=1024,
    )

    build_html(
        Path(args.output),
        {"DINOv2": dinov2_cache, "SigLIP": siglip_cache},
        image_paths,
        args.top_k,
    )

    log("Done. Open the HTML file in your browser to compare.")


if __name__ == "__main__":
    main()
