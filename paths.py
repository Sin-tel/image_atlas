from pathlib import Path


BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

DB_PATH = CACHE_DIR / "metadata.db"
EMBED_CACHE_PATH = CACHE_DIR / "embeddings.npz"
LAYOUT_CACHE_PATH = CACHE_DIR / "umap_layout.npz"
