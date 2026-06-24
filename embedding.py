import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

try:
    from config import access_token
except ImportError:
    access_token = None  # type: ignore[assignment]

from paths import EMBED_CACHE_PATH, LAYOUT_CACHE_PATH
from util import log, file_cache_key


SIGLIP_MODEL_ID = "google/siglip2-base-patch16-256"
EMBED_DIM = 768
BATCH_SIZE = 64


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


def load_model(device):
    from transformers import AutoModel, AutoProcessor

    log(f"Loading SigLIP2 ({SIGLIP_MODEL_ID})")
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID, token=access_token)
    model = (
        AutoModel.from_pretrained(SIGLIP_MODEL_ID, token=access_token).to(device).eval()
    )
    return processor, model


def embed_batch(processor, model, device, images: list) -> np.ndarray:
    import torch

    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.inference_mode():
        out = model.get_image_features(**inputs)
    feats = out.pooler_output
    return feats.float().cpu().numpy()


def run_embedding_pass(image_paths: list[Path], store: EmbeddingStore):
    import torch

    todo = [p for p in image_paths if not store.has(file_cache_key(p))]

    if not todo:
        log(f"All {len(image_paths)} images already embedded")
        return

    log(f"Embedding {len(todo)} new images ({len(image_paths) - len(todo)} cached)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")
    processor, model = load_model(device)

    for i in tqdm(range(0, len(todo), BATCH_SIZE), desc="embedding"):
        batch_paths = todo[i : i + BATCH_SIZE]
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
            vecs = embed_batch(processor, model, device, imgs)  # type: ignore[var-annotated]
        except Exception as e:
            log(f"  batch failed ({e}), trying one-by-one.")
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
        torch.cuda.empty_cache()


# UMAP layout


def compute_umap_layout(
    store: EmbeddingStore, path_to_id: dict[str, str]
) -> dict[str, tuple[float, float]]:
    """Returns {db_id: (x, y)} in [0, 1] range."""

    # Only lay out images that are both embedded and in the DB
    common = [
        (p, db_id)
        for p, db_id in path_to_id.items()
        if p in {store.idx_to_path[i] for i in range(len(store.cache_key_to_idx))}
    ]

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
            return {
                str(db_id): (float(x), float(y))
                for db_id, (x, y) in zip(cached["db_ids"], coords)
            }

    log(f"Computing UMAP layout for {len(embed_indices)} images...")
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

    # Normalize to [0, 1]
    for axis in range(2):
        lo, hi = coords_raw[:, axis].min(), coords_raw[:, axis].max()
        coords_raw[:, axis] = (coords_raw[:, axis] - lo) / max(hi - lo, 1e-6)

    np.savez_compressed(
        LAYOUT_CACHE_PATH,
        db_ids=np.array(db_ids, dtype=object),
        coords=coords_raw.astype(np.float32),
    )
    log("UMAP layout cached")
    return {
        str(db_id): (float(x), float(y)) for db_id, (x, y) in zip(db_ids, coords_raw)
    }
