"""
imagepile — local image exploration server
==========================================

Serves a UMAP-based visual map of your image collection, with
click-to-explore similarity search. All local, no cloud.

Usage:
    python server.py --paths "D:/Art" "D:/Screenshots" [--recursive]

First run: scans folders, embeds images, computes UMAP layout.
Subsequent runs: loads from cache, only re-embeds new/changed files.

Dependencies:
    pip install fastapi uvicorn[standard] pillow numpy tqdm umap-learn
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers
"""

import argparse
import hashlib
import io
import json
import mimetypes
import os
import sqlite3
import time
import random
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from config import access_token

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

DB_PATH = CACHE_DIR / "metadata.db"
EMBED_CACHE_PATH = CACHE_DIR / "embeddings.npz"
LAYOUT_CACHE_PATH = CACHE_DIR / "umap_layout.npz"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

SIGLIP_MODEL_ID = "google/siglip2-base-patch16-256"
EMBED_DIM = 768
BATCH_SIZE = 64
TOP_K = 50  # neighbors returned per query (client decides how many to show)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("PRAGMA foreign_keys = ON")

    con.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id       TEXT PRIMARY KEY,
            path     TEXT UNIQUE NOT NULL,
            added_at REAL NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            image_id TEXT PRIMARY KEY REFERENCES images(id) ON UPDATE CASCADE ON DELETE CASCADE,
            tags     TEXT DEFAULT '',
            source   TEXT DEFAULT '',
            comment  TEXT DEFAULT ''
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_images_path ON images(path)")
    con.commit()

    return con

# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

def file_cache_key(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}|{stat.st_mtime}|{stat.st_size}"
    return hashlib.sha1(raw.encode()).hexdigest()

def find_images(paths: list[Path], recursive: bool) -> list[Path]:
    result = []
    for root in paths:
        pattern = "**/*" if recursive else "*"
        result.extend(
            p for p in root.glob(pattern)
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    return sorted(set(result))

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class EmbeddingStore:
    """In-memory store, persisted to .npz. Keyed by cache_key."""

    def __init__(self):
        self.cache_key_to_idx: dict[str, int] = {}
        self.idx_to_path: dict[int, str] = {}
        self.matrix: np.ndarray = np.zeros((0, EMBED_DIM), dtype=np.float32)
        self._dirty = False
        self._load()

    def _load(self):
        if not EMBED_CACHE_PATH.exists():
            return
        data = np.load(EMBED_CACHE_PATH, allow_pickle=True)
        keys = data["keys"].tolist()
        paths = data["paths"].tolist()
        vecs = data["vectors"]
        self.matrix = vecs
        self.cache_key_to_idx = {k: i for i, k in enumerate(keys)}
        self.idx_to_path = {i: p for i, p in enumerate(paths)}
        log(f"Loaded {len(keys)} cached embeddings")

    def save(self):
        if not self._dirty:
            return
        n = len(self.cache_key_to_idx)
        keys = [""] * n
        paths = [""] * n
        for k, i in self.cache_key_to_idx.items():
            keys[i] = k
            paths[i] = self.idx_to_path[i]
        np.savez_compressed(
            EMBED_CACHE_PATH,
            vectors=self.matrix[:n],
            keys=np.array(keys, dtype=object),
            paths=np.array(paths, dtype=object),
        )
        self._dirty = False
        log(f"Saved {n} embeddings to cache")

    def has(self, cache_key: str) -> bool:
        return cache_key in self.cache_key_to_idx

    def add(self, cache_key: str, path: str, vec: np.ndarray):
        idx = len(self.cache_key_to_idx)
        self.cache_key_to_idx[cache_key] = idx
        self.idx_to_path[idx] = path
        if idx >= self.matrix.shape[0]:
            grown = np.zeros((max(idx * 2, 256), EMBED_DIM), dtype=np.float32)
            grown[: self.matrix.shape[0]] = self.matrix
            self.matrix = grown
        self.matrix[idx] = vec
        self._dirty = True

    def get_normalized_matrix(self, indices: list[int]) -> np.ndarray:
        mat = self.matrix[indices]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return mat / norms

    def query(self, cache_key: str, k: int) -> list[int]:
        """Return indices of top-k most similar images (excluding self)."""
        if cache_key not in self.cache_key_to_idx:
            return []
        q_idx = self.cache_key_to_idx[cache_key]
        n = len(self.cache_key_to_idx)
        mat = self.get_normalized_matrix(list(range(n)))
        q = mat[q_idx]
        sims = mat @ q
        sims[q_idx] = -np.inf
        top = np.argpartition(-sims, kth=min(k, n - 2))[:k]
        top = top[np.argsort(-sims[top])]
        return top.tolist()


def load_siglip_model(device):
    from transformers import AutoModel, AutoProcessor
    log(f"Loading SigLIP2 ({SIGLIP_MODEL_ID}) ...")
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID, token=access_token)
    model = AutoModel.from_pretrained(SIGLIP_MODEL_ID, token=access_token).to(device).eval()
    return processor, model


def embed_batch(processor, model, device, images: list) -> np.ndarray:
    import torch
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        feats = model.get_image_features(**inputs)
    # get_image_features returns a tensor directly for SigLIP2
    if hasattr(feats, "pooler_output"):
        feats = feats.pooler_output
    return feats.float().cpu().numpy()


def run_embedding_pass(image_paths: list[Path], store: EmbeddingStore):
    import torch
    from tqdm import tqdm

    todo = [p for p in image_paths if not store.has(file_cache_key(p))]

    if not todo:
        log(f"All {len(image_paths)} images already embedded")
        return

    log(f"Embedding {len(todo)} new images ({len(image_paths) - len(todo)} cached) ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")
    processor, model = load_siglip_model(device)

    for i in tqdm(range(0, len(todo), BATCH_SIZE), desc="embedding"):
        batch_paths = todo[i: i + BATCH_SIZE]
        imgs, valid = [], []
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                valid.append(p)
            except Exception as e:
                log(f"  skip: {p.name} ({e})")

        if not imgs:
            continue

        try:
            vecs = embed_batch(processor, model, device, imgs)
        except Exception as e:
            log(f"  batch failed ({e}), trying one-by-one ...")
            vecs = []
            for img in imgs:
                try:
                    vecs.append(embed_batch(processor, model, device, [img])[0])
                except Exception as e2:
                    log(f"    failed: {e2}")
                    vecs.append(np.zeros(EMBED_DIM, dtype=np.float32))

        for p, v in zip(valid, vecs):
            store.add(file_cache_key(p), str(p), np.asarray(v, dtype=np.float32))

    store.save()
    del model
    if torch.cuda.is_available():
        import torch
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# UMAP layout
# ---------------------------------------------------------------------------

def compute_umap_layout(store: EmbeddingStore, path_to_id: dict[str, str]) -> dict[str, tuple[float, float]]:
    """Returns {db_id: (x, y)} in [0, 1] range."""

    # Only lay out images that are both embedded and in the DB
    common = [(p, db_id) for p, db_id in path_to_id.items()
              if p in {store.idx_to_path[i] for i in range(len(store.cache_key_to_idx))}]

    # Build path->embed_idx lookup
    path_to_embed_idx = {v: k for k, v in store.idx_to_path.items()}
    common = [(p, db_id) for p, db_id in common if p in path_to_embed_idx]

    if not common:
        return {}

    paths, db_ids = zip(*common)
    embed_indices = [path_to_embed_idx[p] for p in paths]

    # Check if we have a valid cached layout that covers the same set
    if LAYOUT_CACHE_PATH.exists():
        cached = np.load(LAYOUT_CACHE_PATH, allow_pickle=True)
        cached_ids = set(cached["db_ids"].tolist())
        if cached_ids == set(db_ids):
            log("UMAP layout loaded from cache")
            coords = cached["coords"]
            return {str(db_id): (float(x), float(y))
                    for db_id, (x, y) in zip(cached["db_ids"], coords)}

    log(f"Computing UMAP layout for {len(embed_indices)} images (this takes a minute or two) ...")
    import umap
    mat = store.get_normalized_matrix(embed_indices)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=20,
        min_dist=0.3,
        metric="cosine",
        random_state=42,
        verbose=True,
    )
    coords_raw = reducer.fit_transform(mat)

    # Normalize to [0.05, 0.95] so points aren't right at the edges
    for axis in range(2):
        lo, hi = coords_raw[:, axis].min(), coords_raw[:, axis].max()
        coords_raw[:, axis] = 0.05 + 0.9 * (coords_raw[:, axis] - lo) / max(hi - lo, 1e-6)

    np.savez_compressed(
        LAYOUT_CACHE_PATH,
        db_ids=np.array(db_ids, dtype=object),
        coords=coords_raw.astype(np.float32),
    )
    log("UMAP layout cached")
    return {str(db_id): (float(x), float(y))
            for db_id, (x, y) in zip(db_ids, coords_raw)}


# ---------------------------------------------------------------------------
# Thumbnail helper
# ---------------------------------------------------------------------------

def make_thumbnail(path: Path, size: int) -> bytes:
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82, optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# App state (populated at startup)
# ---------------------------------------------------------------------------

class AppState:
    db: sqlite3.Connection = None
    store: EmbeddingStore = None
    layout: dict[str, tuple[float, float]] = {}  # db_id -> (x, y)
    id_to_path: dict[str, str] = {}
    embed_idx_to_db_id: dict[int, str] = {}


state = AppState()
app = FastAPI()

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/layout")
def api_layout():
    """Return UMAP coordinates for all images as a compact list."""
    rows = state.db.execute(
        "SELECT id, path FROM images"
    ).fetchall()

    items = []
    for db_id, path in rows:
        if db_id not in state.layout:
            continue
        x, y = state.layout[db_id]
        items.append({"id": db_id, "x": round(x, 5), "y": round(y, 5)})

    return JSONResponse({"items": items})


@app.get("/api/similar/{image_id}")
def api_similar(image_id: str, k: int = Query(default=50, le=200)):
    if image_id not in state.id_to_path:
        raise HTTPException(404, "Image not found")

    neighbor_indices = state.store.query(image_id, k)

    result = []
    for idx in neighbor_indices:
        db_id = state.embed_idx_to_db_id.get(idx)
        if db_id is not None:
            result.append(db_id)

    return JSONResponse({"ids": result})


@app.get("/api/image/{image_id}/info")
def api_image_info(image_id: str):
    row = state.db.execute("""
        SELECT i.path, i.added_at, COALESCE(m.tags,''), COALESCE(m.source,''), COALESCE(m.comment,'')
        FROM images i LEFT JOIN metadata m ON m.image_id = i.id
        WHERE i.id = ?
    """, (image_id,)).fetchone()
    if not row:
        raise HTTPException(404)
    path, added_at, tags, source, comment = row
    p = Path(path)
    return JSONResponse({
        "id": image_id,
        "filename": p.name,
        "path": path,
        "folder": str(p.parent),
        "added_at": added_at,
        "tags": tags,
        "source": source,
        "comment": comment,
    })


@app.post("/api/image/{image_id}/meta")
async def api_update_meta(image_id: str, request_body: dict):
    tags = request_body.get("tags", "")
    source = request_body.get("source", "")
    comment = request_body.get("comment", "")
    state.db.execute("""
        INSERT INTO metadata (image_id, tags, source, comment)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(image_id) DO UPDATE SET
            tags = excluded.tags,
            source = excluded.source,
            comment = excluded.comment
    """, (image_id, tags, source, comment))
    state.db.commit()
    return JSONResponse({"ok": True})


@app.get("/api/thumbnail/{image_id}")
def api_thumbnail(image_id: str, size: int = Query(default=200, le=800)):
    row = state.db.execute("SELECT path FROM images WHERE id = ?", (image_id,)).fetchone()
    if not row:
        raise HTTPException(404)
    path = Path(row[0])
    if not path.exists():
        raise HTTPException(404, "File not on disk")
    try:
        data = make_thumbnail(path, size)
        return Response(content=data, media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=86400"})
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/image/{image_id}/full")
def api_full_image(image_id: str):
    row = state.db.execute("SELECT path FROM images WHERE id = ?", (image_id,)).fetchone()
    if not row:
        raise HTTPException(404)
    path = Path(row[0])
    if not path.exists():
        raise HTTPException(404, "File not on disk")
    mt = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(str(path), media_type=mt,
                        headers={"Cache-Control": "max-age=3600"})


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def startup(scan_paths: list[Path], recursive: bool):
    log("=== imagepile startup ===")
    state.db = init_db()
    state.store = EmbeddingStore()

    # Scan for images
    log("Scanning for images ...")
    image_paths = find_images(scan_paths, recursive)
    log(f"Found {len(image_paths)} images")

    # Embed new images
    run_embedding_pass(image_paths, state.store)

    # Sync DB
    log("Syncing database ...")
    db_state = {path: db_id for db_id, path in state.db.execute("SELECT id, path FROM images").fetchall()}

    new_count = 0
    update_count = 0

    for p in image_paths:
        p_str = str(p)
        ck = file_cache_key(p)

        if p_str not in db_state:
            # File is completely new
            state.db.execute(
                "INSERT INTO images (id, path, added_at) VALUES (?, ?, ?)",
                (ck, p_str, time.time())
            )
            new_count += 1
        elif db_state[p_str] != ck:
            # File is at the same path but content has been modified
            old_id = db_state[p_str]
            state.db.execute("UPDATE metadata SET image_id = ? WHERE image_id = ?", (ck, old_id))
            state.db.execute("UPDATE images SET id = ? WHERE path = ?", (ck, p_str))
            update_count += 1

    # Optional: clean up missing files
    current_paths = {str(p) for p in image_paths}
    deleted_paths = set(db_state.keys()) - current_paths
    if deleted_paths:
        for p_str in deleted_paths:
            state.db.execute("DELETE FROM images WHERE path = ?", (p_str,))
        log(f"Removed {len(deleted_paths)} missing images from DB")

    if new_count or update_count or deleted_paths:
        state.db.commit()
        log(f"Database sync: {new_count} new, {update_count} updated.")

    # Build lookup dicts
    for db_id, path in state.db.execute("SELECT id, path FROM images").fetchall():
        state.id_to_path[db_id] = path

    path_to_id = {path: db_id for db_id, path in state.id_to_path.items()}

    # Compute reverse embed lookup explicitly once at startup
    state.embed_idx_to_db_id = {
        state.store.cache_key_to_idx[db_id]: db_id
        for db_id in state.id_to_path
        if db_id in state.store.cache_key_to_idx
    }

    # UMAP layout
    state.layout = compute_umap_layout(state.store, path_to_id)
    log(f"Layout ready for {len(state.layout)} images")
    log("=== Ready — open http://localhost:8765 ===")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    scan_paths = [Path(p) for p in args.paths]
    for p in scan_paths:
        if not p.exists():
            print(f"ERROR: path does not exist: {p}")
            raise SystemExit(1)

    startup(scan_paths, args.recursive)

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
