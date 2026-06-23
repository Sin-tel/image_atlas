const MAP_THUMB_SIZE = 32;
const ZOOM_THUMB_MIN = 1.2;
const PREFETCH_MARGIN = 1.5;
const LRU_MAX = 2500;

const canvas = document.getElementById('map-canvas');
const ctx = canvas.getContext('2d');
const wrap = document.getElementById('canvas-wrap');
const status = document.getElementById('status');
const btnMap = document.getElementById('btn-map');

let items = [];
let viewMode = 'map';

let masonryItems = [];
let masonryMain = null;
let resizeTimer = null;

let vp = { tx: 0, ty: 0, scale: 1 };

// Cache

const imgCache = new Map();
const lruOrder = [];

function lruGet(id) { return imgCache.get(id); }

function lruSet(id, val) {
    if (imgCache.has(id)) {
        lruOrder.splice(lruOrder.indexOf(id), 1);
    } else if (imgCache.size >= LRU_MAX) {
        const evict = lruOrder.shift();
        imgCache.delete(evict);
    }
    imgCache.set(id, val);
    lruOrder.push(id);
}

// Init
(async function boot() {
    const res = await fetch('/api/layout');
    const data = await res.json();
    items = data.items;
    status.textContent = `${items.length.toLocaleString()} images`;

    resizeCanvas();
    fitAll();
    renderLoop();
    status.textContent = `${items.length.toLocaleString()} images — scroll/pinch to zoom, click to explore`;
})();

// Canvas sizing & Event listeners

function resizeCanvas() {
    canvas.width = wrap.clientWidth;
    canvas.height = wrap.clientHeight;
}

window.addEventListener('resize', () => {
    resizeCanvas();
    if (viewMode === 'map') scheduleDraw();
    if (viewMode === 'neighbors') {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(layoutMasonry, 50);
    }
});

// Coordinate helpers

const worldW = () => canvas.width * 3;
const worldH = () => canvas.height * 3;

function worldToScreen(wx, wy) {
    return {
        sx: wx * worldW() * vp.scale + vp.tx,
        sy: wy * worldH() * vp.scale + vp.ty,
    };
}

function screenToWorld(sx, sy) {
    return {
        wx: (sx - vp.tx) / (worldW() * vp.scale),
        wy: (sy - vp.ty) / (worldH() * vp.scale),
    };
}

function fitAll() {
    const pad = 0.05;
    vp.scale = 1 - pad * 2;
    vp.tx = canvas.width * pad;
    vp.ty = canvas.height * pad;
}

// Render loop

let rafId = null;
let needsDraw = true;

function renderLoop() {
    if (needsDraw) { draw();
        needsDraw = false; }
    rafId = requestAnimationFrame(renderLoop);
}

function scheduleDraw() { needsDraw = true; }

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const thumbHalf = (MAP_THUMB_SIZE / 2) * vp.scale;
    const showThumbs = vp.scale >= ZOOM_THUMB_MIN;

    const tl = screenToWorld(-thumbHalf * PREFETCH_MARGIN, -thumbHalf * PREFETCH_MARGIN);
    const br = screenToWorld(canvas.width + thumbHalf * PREFETCH_MARGIN,
        canvas.height + thumbHalf * PREFETCH_MARGIN);

    const visible = items.filter(it =>
        it.x >= tl.wx && it.x <= br.wx &&
        it.y >= tl.wy && it.y <= br.wy
    );

    for (const it of visible) {
        const { sx, sy } = worldToScreen(it.x, it.y);
        const inViewport = sx >= -thumbHalf && sx <= canvas.width + thumbHalf &&
            sy >= -thumbHalf && sy <= canvas.height + thumbHalf;

        if (showThumbs) {
            const img = lruGet(it.id);

            if (!img) {
                startLoad(it.id);
                drawDot(sx, sy, 4, '#3f3f50');
            } else if (img === 'loading') {
                drawDot(sx, sy, 4, '#4f4f60');
            } else if (img === 'error') {
                drawDot(sx, sy, 3, '#7f2020');
            } else {
                if (inViewport) drawThumb(img, sx, sy, thumbHalf * 2);
            }
        } else {
            drawDot(sx, sy, 5 * vp.scale, '#6366f1');
        }
    }
}

function drawDot(sx, sy, r, color) {
    ctx.beginPath();
    ctx.arc(sx, sy, Math.max(r, 1.5), 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
}

function drawThumb(img, sx, sy, size) {
    const half = size / 2;
    const r = Math.max(2, 4 * vp.scale);
    ctx.save();
    ctx.beginPath();
    ctx.roundRect(sx - half, sy - half, size, size, r);
    ctx.clip();
    ctx.drawImage(img, sx - half, sy - half, size, size);
    ctx.restore();
}

function startLoad(id) {
    if (imgCache.has(id)) return;
    lruSet(id, 'loading');
    const img = new Image();
    img.onload = () => { lruSet(id, img);
        scheduleDraw(); };
    img.onerror = () => { lruSet(id, 'error'); };
    img.src = `/api/thumbnail/${id}?size=200`;
}

// Pan & zoom

let drag = null;

wrap.addEventListener('mousedown', e => {
    if (e.button !== 0) return;
    drag = { startX: e.clientX, startY: e.clientY, tx0: vp.tx, ty0: vp.ty, moved: false };
    wrap.classList.add('grabbing');
    e.preventDefault();
});

wrap.addEventListener('mousemove', e => {
    if (!drag) return;
    const dx = e.clientX - drag.startX;
    const dy = e.clientY - drag.startY;
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) drag.moved = true;
    vp.tx = drag.tx0 + dx;
    vp.ty = drag.ty0 + dy;
    scheduleDraw();
});

wrap.addEventListener('mouseup', e => {
    if (!drag) return;
    const moved = drag.moved;
    drag = null;
    wrap.classList.remove('grabbing');
    if (!moved) handleMapClick(e.clientX, e.clientY);
});

wrap.addEventListener('mouseleave', () => {
    drag = null;
    wrap.classList.remove('grabbing');
});

wrap.addEventListener('wheel', e => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    const cx = e.clientX,
        cy = e.clientY;
    vp.tx = cx - (cx - vp.tx) * factor;
    vp.ty = cy - (cy - vp.ty) * factor;
    vp.scale *= factor;
    scheduleDraw();
}, { passive: false });

let lastPinchDist = null;
wrap.addEventListener('touchstart', e => {
    if (e.touches.length === 2) lastPinchDist = pinchDist(e);
}, { passive: true });
wrap.addEventListener('touchmove', e => {
    if (e.touches.length !== 2) return;
    const d = pinchDist(e);
    const factor = d / lastPinchDist;
    const cx = (e.touches[0].clientX + e.touches[1].clientX) / 2;
    const cy = (e.touches[0].clientY + e.touches[1].clientY) / 2;
    vp.tx = cx - (cx - vp.tx) * factor;
    vp.ty = cy - (cy - vp.ty) * factor;
    vp.scale *= factor;
    lastPinchDist = d;
    scheduleDraw();
}, { passive: true });

function pinchDist(e) {
    const dx = e.touches[0].clientX - e.touches[1].clientX;
    const dy = e.touches[0].clientY - e.touches[1].clientY;
    return Math.hypot(dx, dy);
}

// Hit testing & Navigation

function hitTest(sx, sy) {
    const thumbHalf = (MAP_THUMB_SIZE / 2) * vp.scale;
    const tl = screenToWorld(sx - thumbHalf, sy - thumbHalf);
    const br = screenToWorld(sx + thumbHalf, sy + thumbHalf);

    let best = null,
        bestDist = Infinity;
    for (const it of items) {
        if (it.x < tl.wx || it.x > br.wx || it.y < tl.wy || it.y > br.wy) continue;
        const { sx: cx, sy: cy } = worldToScreen(it.x, it.y);
        const d = Math.hypot(sx - cx, sy - cy);
        if (d < thumbHalf && d < bestDist) { best = it;
            bestDist = d; }
    }
    return best;
}

function handleMapClick(sx, sy) {
    const hit = hitTest(sx, sy);
    if (hit) showNeighbors(hit.id);
}

function switchView(mode) {
    viewMode = mode;
    if (mode === 'map') {
        wrap.style.display = 'block';
        document.getElementById('neighbors-view').classList.remove('active');
        btnMap.style.display = 'none';
        scheduleDraw();
    } else {
        wrap.style.display = 'none';
        document.getElementById('neighbors-view').classList.add('active');
        btnMap.style.display = 'block';
    }
}
btnMap.addEventListener('click', () => switchView('map'));

// ============================================================
// Neighbors view (Masonry Layout)
// ============================================================

const COLS = 8;
const GAP = 16;

function layoutMasonry() {
    const container = document.getElementById('masonry-container');
    if (!container || !masonryMain) return;

    const width = container.clientWidth;
    if (!width) return;

    const colWidth = (width - GAP * (COLS - 1)) / COLS;
    let heights = new Array(COLS).fill(0);

    // 1. Layout Main Image (Spans 4 columns)
    const mainEl = document.getElementById(`nb-${masonryMain.id}`);
    if (mainEl) {
        const mainW = colWidth * 4 + GAP * 3;
        const mainH = masonryMain.height * (mainW / masonryMain.width);

        mainEl.style.width = `${mainW}px`;
        mainEl.style.height = `${mainH}px`;
        mainEl.style.transform = `translate(0px, 0px)`;

        // Fill the heights for the first 4 columns
        for (let i = 0; i < 4; i++) {
            heights[i] = mainH + GAP;
        }
    }

    // 2. Layout similar images into the shortest column
    for (const item of masonryItems) {
        const el = document.getElementById(`nb-${item.id}`);
        if (!el) continue;

        let minCol = 0;
        let minH = heights[0];
        for (let i = 1; i < COLS; i++) {
            if (heights[i] < minH) {
                minH = heights[i];
                minCol = i;
            }
        }

        const itemH = item.height * (colWidth / item.width);
        const x = minCol * (colWidth + GAP);
        const y = minH;

        el.style.width = `${colWidth}px`;
        el.style.height = `${itemH}px`;
        el.style.transform = `translate(${x}px, ${y}px)`;

        heights[minCol] += itemH + GAP;
    }

    container.style.height = `${Math.max(...heights)}px`;
}

async function showNeighbors(id) {
    switchView('neighbors');

    const container = document.getElementById('masonry-container');
    container.innerHTML = '';
    document.getElementById('neighbors-view').scrollTop = 0;

    const [info, simRes] = await Promise.all([
        fetch(`/api/image/${id}/info`).then(r => r.json()),
        fetch(`/api/similar/${id}`).then(r => r.json()),
    ]);

    masonryMain = info;
    masonryItems = simRes.items;

    // Create Main element
    const mainEl = document.createElement('div');
    mainEl.id = `nb-${info.id}`;
    mainEl.className = 'masonry-item masonry-main';
    mainEl.title = 'Click to view full resolution';
    mainEl.innerHTML = `<img src="/api/thumbnail/${info.id}?size=1000" alt="">`;
    mainEl.addEventListener('click', () => window.open(`/api/image/${info.id}/full`, '_blank'));
    container.appendChild(mainEl);

    // Create similar items
    masonryItems.forEach((item, index) => {
        const el = document.createElement('div');
        el.id = `nb-${item.id}`;
        el.className = 'masonry-item';
        el.innerHTML = `
      <img loading="lazy" src="/api/thumbnail/${item.id}?size=400" alt="">
      <div class="nb-rank">#${index + 1}</div>
    `;
        el.addEventListener('click', () => showNeighbors(item.id));
        container.appendChild(el);
    });

    layoutMasonry();
}
