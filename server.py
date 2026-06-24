import io
import mimetypes
import sqlite3
import time
from pathlib import Path
import webbrowser
from tqdm import tqdm

import uvicorn
from PIL import Image
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles


from paths import DB_PATH, BASE_DIR
from util import log, find_images, file_cache_key, get_image_metadata
from embedding import EmbeddingStore, run_embedding_pass, compute_umap_layout
from config import scan_folders


def init_db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id       TEXT PRIMARY KEY,
            path     TEXT UNIQUE NOT NULL,
            width    INTEGER NOT NULL,
            height   INTEGER NOT NULL,
            color    TEXT NOT NULL,
            added_at REAL NOT NULL
        )
    """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_images_path ON images(path)")
    con.commit()

    return con


def make_thumbnail(path: Path, size: int) -> bytes:
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82, optimize=True)
    return buf.getvalue()


# App state


class AppState:
    db: sqlite3.Connection = None  # type: ignore[assignment]
    store: EmbeddingStore = None  # type: ignore[assignment]
    layout: dict[str, tuple[float, float]] = {}  # db_id -> (x, y)
    id_to_path: dict[str, str] = {}
    embed_idx_to_db_id: dict[int, str] = {}


state = AppState()
app = FastAPI()


# API routes
@app.get("/api/layout")
def api_layout():
    rows = state.db.execute("SELECT id, path, color FROM images").fetchall()
    items = []
    for db_id, path, color in rows:
        if db_id not in state.layout:
            continue
        x, y = state.layout[db_id]
        items.append({"id": db_id, "x": round(x, 5), "y": round(y, 5), "color": color})

    return JSONResponse({"items": items})


@app.get("/api/similar/{image_id}")
def api_similar(image_id: str, k: int = Query(default=80, le=200)):
    if image_id not in state.id_to_path:
        raise HTTPException(404, "Image not found")

    neighbor_indices = state.store.query(image_id, k)
    db_ids = [
        state.embed_idx_to_db_id.get(idx)
        for idx in neighbor_indices
        if state.embed_idx_to_db_id.get(idx) is not None
    ]

    if not db_ids:
        return JSONResponse({"items": []})

    placeholders = ",".join("?" for _ in db_ids)
    rows = state.db.execute(
        f"SELECT id, width, height FROM images WHERE id IN ({placeholders})", db_ids
    ).fetchall()
    dim_map = {r[0]: (r[1], r[2]) for r in rows}

    result = []
    for db_id in db_ids:
        w, h = dim_map.get(db_id, (100, 100))
        result.append({"id": db_id, "width": w, "height": h})

    return JSONResponse({"items": result})


@app.get("/api/image/{image_id}/info")
def api_image_info(image_id: str):
    row = state.db.execute(
        "SELECT path, width, height FROM images WHERE id = ?", (image_id,)
    ).fetchone()
    if not row:
        raise HTTPException(404)
    path, width, height = row
    return JSONResponse({"id": image_id, "width": width, "height": height})


@app.get("/api/thumbnail/{image_id}")
def api_thumbnail(image_id: str, size: int = Query(default=200, le=1000)):
    row = state.db.execute(
        "SELECT path FROM images WHERE id = ?", (image_id,)
    ).fetchone()
    if not row:
        raise HTTPException(404)
    path = Path(row[0])
    if not path.exists():
        raise HTTPException(404, "File not on disk")
    try:
        data = make_thumbnail(path, size)
        return Response(
            content=data,
            media_type="image/jpeg",
            headers={"Cache-Control": "max-age=86400"},
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/image/{image_id}/full")
def api_full_image(image_id: str):
    row = state.db.execute(
        "SELECT path FROM images WHERE id = ?", (image_id,)
    ).fetchone()
    if not row:
        raise HTTPException(404)
    path = Path(row[0])
    if not path.exists():
        raise HTTPException(404, "File not on disk")
    mt = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(
        str(path), media_type=mt, headers={"Cache-Control": "max-age=3600"}
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


def main():
    scan_paths = [Path(p) for p in scan_folders]
    recursive = True

    log("=== imagepile startup ===")
    state.db = init_db()
    state.store = EmbeddingStore()

    # Scan for images
    log("Scanning for images...")
    image_paths = find_images(scan_paths, recursive)
    log(f"Found {len(image_paths)} images")

    # Embed new images
    run_embedding_pass(image_paths, state.store)

    # Sync DB
    log("Syncing database...")
    db_state = {
        path: db_id
        for db_id, path in state.db.execute("SELECT id, path FROM images").fetchall()
    }

    new_count = 0
    update_count = 0

    for p in tqdm(image_paths):
        p_str = str(p)
        ck = file_cache_key(p)

        if p_str not in db_state:
            w, h, col = get_image_metadata(p)
            state.db.execute(
                "INSERT INTO images (id, path, width, height, color, added_at) VALUES (?, ?, ?, ?, ?, ?)",
                (ck, p_str, w, h, col, time.time()),
            )
            new_count += 1
        elif db_state[p_str] != ck:
            w, h, col = get_image_metadata(p)
            state.db.execute(
                "UPDATE images SET id = ?, width = ?, height = ?, color = ? WHERE path = ?",
                (ck, w, h, col, p_str),
            )
            update_count += 1

    # Clean up missing files
    current_paths = {str(p) for p in image_paths}
    deleted_paths = set(db_state.keys()) - current_paths
    for p_str in deleted_paths:
        state.db.execute("DELETE FROM images WHERE path = ?", (p_str,))
    if deleted_paths:
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

    log("open http://localhost:8765")
    webbrowser.open("http://localhost:8765")

    # Start app
    app.mount("/static", StaticFiles(directory="static", html=True), name="static")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")


if __name__ == "__main__":
    main()
