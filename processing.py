"""Image-processing functions for Snap Remix.

Every function takes and returns OpenCV-style images (BGR / BGRA, uint8).
"""

import cv2
import numpy as np

MAX_WIDTH = 1600  # big phone photos are shrunk so sliders stay fast


# ---------------------------------------------------------------- loading
def load_image(file_bytes):
    """Decode uploaded bytes into a BGR image (or None if it isn't an image)."""
    data = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > MAX_WIDTH:
        new_h = int(h * MAX_WIDTH / w)
        img = cv2.resize(img, (MAX_WIDTH, new_h), interpolation=cv2.INTER_AREA)
    return img


def load_pdf_page(file_bytes, page_number=1):
    """Render one page of a PDF to a BGR image. Returns (image, page_count)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(file_bytes)
    count = len(pdf)
    page = pdf[min(max(page_number, 1), count) - 1]
    w, h = page.get_width(), page.get_height()
    scale = min(MAX_WIDTH / w, 2200 / h, 3.0)  # sharp, but not huge
    rgb = np.array(page.render(scale=scale).to_pil().convert("RGB"))
    pdf.close()
    return np.ascontiguousarray(rgb[:, :, ::-1]), count


# ------------------------------------------------------ image arithmetic
def adjust(img, brightness, contrast, clip=True):
    """Brightness (cv2.add / cv2.subtract) then contrast (cv2.multiply).

    clip=True  -> values are kept in 0-255 (cv2.add saturates, np.clip on contrast)
    clip=False -> plain uint8 math, so values wrap around (250 + 30 = 24)
    """
    if clip:
        matrix = np.ones(img.shape, dtype=np.uint8) * abs(int(brightness))
        if brightness >= 0:
            out = cv2.add(img, matrix)
        else:
            out = cv2.subtract(img, matrix)
        matrix_ones = np.ones(img.shape, dtype="float64")
        out = cv2.multiply(np.float64(out), matrix_ones, scale=contrast)
        return np.uint8(np.clip(out, 0, 255))

    # No overflow fix: let the numbers wrap the way raw uint8 math does
    out = img.astype(np.int64) + int(brightness)
    out = (out * contrast).astype(np.int64)
    return (out % 256).astype(np.uint8)


# ------------------------------------------------------------ thresholding
def make_mask(gray, method, thresh=127, keep_dark=False, block_size=31, c=7):
    """Return (mask, threshold_used). White (255) = keep, black (0) = background."""
    ttype = cv2.THRESH_BINARY_INV if keep_dark else cv2.THRESH_BINARY

    if method == "Global":
        used, mask = cv2.threshold(gray, thresh, 255, ttype)
    elif method == "Otsu (automatic)":
        used, mask = cv2.threshold(gray, 0, 255, ttype + cv2.THRESH_OTSU)
    elif method == "Adaptive":
        block_size = max(3, int(block_size) | 1)  # must be odd
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     ttype, block_size, c)
        used = None
    else:
        raise ValueError(f"Unknown method: {method}")
    return mask, used


# ------------------------------------------------------- alpha channel
def build_alpha(mask, background_opacity=0):
    """Turn a keep-mask into an alpha channel.

    Kept pixels -> 255 (solid). Background pixels -> background_opacity
    (0 = fully transparent, 128 = half see-through).
    Same idea as the kangaroo cell: scale the background, then bitwise_not.
    """
    background = cv2.bitwise_not(mask)  # 255 where we want see-through
    scaled = (background.astype(np.float64) / 255.0 * (255 - background_opacity))
    return cv2.bitwise_not(scaled.astype(np.uint8))


def attach_alpha(img, alpha):
    b, g, r = cv2.split(img)
    return cv2.merge([b, g, r, alpha])


# ------------------------------------------------------------- drawing
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def draw_caption(img, alpha, text, color_bgr, position="Bottom", size=1.0, banner=True):
    """Centered caption with an optional filled banner behind it."""
    if not text.strip():
        return
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = w / 900 * size
    thickness = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)

    # Shrink text that would run off the edges
    if tw > w * 0.95:
        scale *= (w * 0.95) / tw
        thickness = max(1, int(round(scale * 2)))
        (tw, th), base = cv2.getTextSize(text, font, scale, thickness)

    pad = int(th * 0.6)
    x = (w - tw) // 2
    y = (th + 2 * pad) - pad if position == "Top" else h - pad - base

    if banner:
        top = 0 if position == "Top" else h - (th + base + 2 * pad)
        bottom = th + base + 2 * pad if position == "Top" else h
        cv2.rectangle(img, (0, top), (w, bottom), (0, 0, 0), thickness=-1)
        if alpha is not None:
            cv2.rectangle(alpha, (0, top), (w, bottom), 255, thickness=-1)

    cv2.putText(img, text, (x, y), font, scale, color_bgr, thickness, cv2.LINE_AA)
    if alpha is not None:
        cv2.putText(alpha, text, (x, y), font, scale, 255, thickness, cv2.LINE_AA)


def draw_box(img, alpha, x_range, y_range, color_bgr, shape="Rectangle"):
    """Highlight a region; ranges are percentages of width / height."""
    h, w = img.shape[:2]
    x1, x2 = int(w * x_range[0] / 100), int(w * x_range[1] / 100)
    y1, y2 = int(h * y_range[0] / 100), int(h * y_range[1] / 100)
    thickness = max(2, w // 300)

    if shape == "Rectangle":
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, thickness, cv2.LINE_AA)
        if alpha is not None:
            cv2.rectangle(alpha, (x1, y1), (x2, y2), 255, thickness, cv2.LINE_AA)
    else:  # Circle that fits the selected region
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        radius = max(1, min(x2 - x1, y2 - y1) // 2)
        cv2.circle(img, center, radius, color_bgr, thickness, cv2.LINE_AA)
        if alpha is not None:
            cv2.circle(alpha, center, radius, 255, thickness, cv2.LINE_AA)


# --------------------------------------------------- AI colorization (Lab)
# The three models from the homework notebook. All of them keep the photo's
# lightness (L in Lab) and predict only the two color channels (a, b).
#   ECCV 2016 / SIGGRAPH 2017: Zhang et al., PyTorch (zhang_models.py), 256x256 input
#   DDColor (tiny):            Kang et al. 2023, ONNX file run with onnxruntime, 512x512 input
MODELS = ["ECCV 2016", "SIGGRAPH 2017", "DDColor (tiny)"]
MODEL_INFO = {
    "ECCV 2016": "Zhang et al. 2016: classifies each spot into one of 313 colors. ~123 MB.",
    "SIGGRAPH 2017": "Zhang et al. 2017: a U-Net trained to predict color directly. ~130 MB.",
    "DDColor (tiny)": "Kang et al. 2023 (ICCV): dual decoders; the most realistic of the three. ~135 MB.",
}
DDCOLOR_FILE = "ddcolor-tiny-fp16.onnx"
DDCOLOR_URLS = ["https://huggingface.co/edgetools/ddcolor/resolve/main/ddcolor-tiny-fp16.onnx"]


def cache_dir():
    """Model files live outside the app folder so they never end up in the repo."""
    import os
    path = os.path.join(os.path.expanduser("~"), ".cache", "snap-remix")
    os.makedirs(path, exist_ok=True)
    return path


def download(urls, filename, min_bytes):
    """Download a model file once (trying each mirror) and return its path."""
    import os
    import urllib.request

    path = os.path.join(cache_dir(), filename)
    if os.path.exists(path) and os.path.getsize(path) >= min_bytes:
        return path
    errors = []
    for url in urls:
        tmp = path + ".part"
        try:
            urllib.request.urlretrieve(url, tmp)
            if os.path.getsize(tmp) >= min_bytes:
                os.replace(tmp, path)
                return path
            errors.append(f"{url}: file too small")
        except Exception as exc:
            errors.append(f"{url}: {exc}")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    raise RuntimeError(f"Couldn't download {filename}. " + " | ".join(errors))


def model_available(name):
    """Is the library this model needs installed?"""
    import importlib.util
    lib = "onnxruntime" if name == "DDColor (tiny)" else "torch"
    return importlib.util.find_spec(lib) is not None


def model_downloaded(name):
    import os
    if name == "DDColor (tiny)":
        return os.path.exists(os.path.join(cache_dir(), DDCOLOR_FILE))
    want = "colorization_release_v2" if name == "ECCV 2016" else "siggraph17"
    try:  # only used to pick the spinner message, so never let it fail
        import torch
        ckpt = os.path.join(torch.hub.get_dir(), "checkpoints")
        return os.path.isdir(ckpt) and any(f.startswith(want) for f in os.listdir(ckpt))
    except Exception:
        return False


def load_model(name):
    """Build a model (downloading its weights the first time)."""
    if name == "DDColor (tiny)":
        import onnxruntime as ort
        path = download(DDCOLOR_URLS, DDCOLOR_FILE, 100_000_000)
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    import zhang_models
    model = zhang_models.eccv16() if name == "ECCV 2016" else zhang_models.siggraph17()
    return model.eval()


def bgr_to_L(img):
    """Lightness channel (0-100) of a BGR uint8 image, OpenCV float Lab."""
    rgb = img[:, :, ::-1].astype(np.float32) / 255.0
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[:, :, 0]


def L_ab_to_bgr(L, ab):
    """Join the original full-size L with predicted a, b and go back to BGR uint8."""
    lab = np.concatenate([L[:, :, None], ab.astype(np.float32)], axis=2)
    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    return np.uint8(np.clip(rgb, 0, 1)[:, :, ::-1] * 255 + 0.5)


def predict_ab(name, model, img):
    """Run one model; returns a, b at the model's own resolution (H x W x 2)."""
    if name == "DDColor (tiny)":
        # 512x512, a gray RGB rebuilt from L (a = b = 0), no ImageNet normalization
        small = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        L = bgr_to_L(small)
        gray = cv2.cvtColor(np.dstack([L, np.zeros_like(L), np.zeros_like(L)]), cv2.COLOR_Lab2RGB)
        x = np.ascontiguousarray(gray.transpose(2, 0, 1)[None], dtype=np.float32)
        ab = model.run(None, {model.get_inputs()[0].name: x})[0][0]
        return ab.transpose(1, 2, 0)
    # Zhang models: L of the photo shrunk to 256x256 (like preprocess_img in the notebook)
    import torch
    small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    L = torch.from_numpy(bgr_to_L(small))[None, None]
    with torch.no_grad():
        ab = model(L)[0].cpu().numpy()
    return ab.transpose(1, 2, 0)


def colorize(img, name, model, strength=1.0):
    """Predict a, b at low resolution, resize them up, keep the full-size original L."""
    h, w = img.shape[:2]
    ab = cv2.resize(predict_ab(name, model, img), (w, h), interpolation=cv2.INTER_LINEAR)
    return L_ab_to_bgr(bgr_to_L(img), ab * float(strength))


def comparison_grid(images, titles, height=420):
    """Original + each model's result in one row with a title above each (like the notebook)."""
    tiles = []
    for im, title in zip(images, titles):
        w = int(im.shape[1] * height / im.shape[0])
        tile = cv2.copyMakeBorder(cv2.resize(im, (w, height), interpolation=cv2.INTER_AREA),
                                  48, 12, 12, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(tile, title, ((tile.shape[1] - tw) // 2, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (40, 40, 40), 2, cv2.LINE_AA)
        tiles.append(tile)
    return np.hstack(tiles)


def looks_black_and_white(img, tolerance=12):
    """True when the three channels are (almost) identical."""
    small = cv2.resize(img, (200, int(200 * img.shape[0] / img.shape[1]) or 1))
    spread = small.max(axis=2).astype(int) - small.min(axis=2).astype(int)
    return float(np.percentile(spread, 95)) <= tolerance


# ------------------------------------------------------ equalization & histograms
EQUALIZE = ["Off", "Brightness only (HSV V channel)", "Each color channel (false colors)"]


def equalize(img, mode):
    """Histogram equalization: spread the brightness levels over 0-255."""
    if mode == EQUALIZE[1]:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if mode == EQUALIZE[2]:
        out = np.zeros_like(img)
        for i in range(3):
            out[:, :, i] = cv2.equalizeHist(img[:, :, i])
        return out
    return img


def histograms(img, mask=None):
    """256-bin count per channel (B, G, R) with cv2.calcHist; mask = only white pixels."""
    if mask is not None:
        mask = np.uint8(mask > 0) * 255
    return np.column_stack([
        cv2.calcHist([img], [c], mask, [256], [0, 256]).ravel() for c in range(3)
    ])


def side_by_side(left, right, gap=16):
    """Original and result next to each other on white, for handing in both."""
    def solid(x):
        if x.shape[2] == 4:
            a = x[:, :, 3:4].astype(np.float32) / 255
            return (x[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
        return x
    left, right = solid(left), solid(right)
    right = cv2.resize(right, (int(right.shape[1] * left.shape[0] / right.shape[0]), left.shape[0]))
    h = left.shape[0]
    spacer = np.full((h, gap, 3), 255, np.uint8)
    return np.hstack([left, spacer, right])


# ------------------------------------------------------------- presets
PRESETS = ["None", "Golden Hour", "Noir", "Pop Art", "Glitch", "Scan (clean document)"]


def _channel_shift(img, b=0, g=0, r=0):
    """Add / subtract a constant per channel with saturation (cv2.add / cv2.subtract)."""
    out = []
    for ch, amount in zip(cv2.split(img), (b, g, r)):
        m = np.full(ch.shape, abs(amount), dtype=np.uint8)
        out.append(cv2.add(ch, m) if amount >= 0 else cv2.subtract(ch, m))
    return cv2.merge(out)


def apply_preset(img, preset):
    if preset == "Golden Hour":
        # warmer: more red, less blue, then a gentle contrast boost
        warm = _channel_shift(img, b=-25, g=5, r=35)
        return np.uint8(np.clip(cv2.multiply(np.float64(warm), np.ones(warm.shape), scale=1.1), 0, 255))
    if preset == "Noir":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.uint8(np.clip(cv2.multiply(np.float64(gray), np.ones(gray.shape), scale=1.4) - 40, 0, 255))
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if preset == "Pop Art":
        # posterize: global threshold on every channel separately
        chans = []
        for ch in cv2.split(img):
            _, hi = cv2.threshold(ch, 170, 255, cv2.THRESH_BINARY)
            _, mid = cv2.threshold(ch, 85, 128, cv2.THRESH_BINARY)
            chans.append(cv2.max(hi, mid))
        return cv2.merge(chans)
    if preset == "Glitch":
        # overflow on purpose (no clipping) plus offset color channels
        wrapped = ((img.astype(np.int64) * 1.6) % 256).astype(np.uint8)
        b, g, r = cv2.split(wrapped)
        shift = max(4, img.shape[1] // 120)
        r = np.roll(r, shift, axis=1)
        b = np.roll(b, -shift, axis=1)
        return cv2.merge([b, g, r])
    if preset == "Scan (clean document)":
        # adaptive threshold: crisp black ink on white paper, even with shadows
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        block = max(15, (img.shape[1] // 60) | 1)
        clean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, block, 12)
        return cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
    return img.copy()


# ---------------------------------------------------------- backgrounds
BACKGROUNDS = ["Transparent", "Sunset sky", "Ocean blue", "Starry night",
               "Solid color", "My own image"]


def _vertical_gradient(h, w, stops):
    """stops: list of (position 0-1, (B, G, R)). Returns an h x w BGR image."""
    ys = np.linspace(0, 1, h)
    out = np.zeros((h, 3))
    pos = [s[0] for s in stops]
    for c in range(3):
        out[:, c] = np.interp(ys, pos, [s[1][c] for s in stops])
    return np.repeat(out[:, None, :], w, axis=1).astype(np.uint8)


def make_background(choice, h, w, color_bgr=(0, 0, 0), custom=None):
    if choice == "Sunset sky":
        bg = _vertical_gradient(h, w, [(0, (90, 30, 60)), (0.45, (80, 60, 220)),
                                       (0.75, (60, 150, 255)), (1, (150, 210, 255))])
        cv2.circle(bg, (int(w * 0.72), int(h * 0.62)), max(8, w // 14), (170, 230, 255), -1, cv2.LINE_AA)
        return bg
    if choice == "Ocean blue":
        return _vertical_gradient(h, w, [(0, (235, 200, 120)), (0.5, (210, 210, 60)),
                                         (1, (170, 160, 20))])
    if choice == "Starry night":
        bg = _vertical_gradient(h, w, [(0, (40, 10, 10)), (1, (110, 50, 30))])
        rng = np.random.default_rng(7)  # same stars every time
        for _ in range(int(h * w / 2500)):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            r = int(rng.choice([1, 1, 1, 2, 3]))
            cv2.circle(bg, (x, y), r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(bg, (int(w * 0.2), int(h * 0.18)), max(6, w // 20), (215, 240, 250), -1, cv2.LINE_AA)
        return bg
    if choice == "Solid color":
        return np.full((h, w, 3), color_bgr, dtype=np.uint8)
    if choice == "My own image" and custom is not None:
        return cv2.resize(custom, (w, h), interpolation=cv2.INTER_AREA)
    return None


def limit_to_top(alpha, sky_end_pct):
    """Only let the top part of the photo be replaced (keeps water/wake below)."""
    out = alpha.copy()
    out[int(alpha.shape[0] * sky_end_pct / 100):, :] = 255
    return out


def sky_touching_top(alpha):
    """Only replace background regions that touch the top edge (real sky does).

    Bright patches inside the subject (sunlit rock, highlights) are put back.
    """
    background = np.uint8(alpha < 128)
    count, labels = cv2.connectedComponents(background, connectivity=8)
    top_labels = np.unique(labels[0, :])
    top_labels = top_labels[top_labels != 0]
    is_sky = np.isin(labels, top_labels)
    out = alpha.copy()
    out[(background == 1) & ~is_sky] = 255
    return out


def feather(alpha, amount):
    """Soften the cut-out edge with a Gaussian blur on the alpha channel."""
    if amount <= 0:
        return alpha
    k = int(amount) * 2 + 1
    return cv2.GaussianBlur(alpha, (k, k), 0)


def composite(fg, alpha, bg):
    """Alpha-blend the foreground over a new background."""
    a = alpha[:, :, None].astype(np.float64) / 255.0
    return (fg.astype(np.float64) * a + bg.astype(np.float64) * (1 - a)).astype(np.uint8)


# ------------------------------------------------------------ postcard
def postcard(img, alpha, title, subtitle, accent_bgr=(40, 120, 230), stamp="SNAP"):
    """Add a white postcard border, a title band and a postage stamp."""
    h, w = img.shape[:2]
    border = max(10, w // 40)
    band = max(40, h // 9)
    framed = cv2.copyMakeBorder(img, border, band + border, border, border,
                                cv2.BORDER_CONSTANT, value=(250, 250, 250))
    new_alpha = None
    if alpha is not None:
        new_alpha = cv2.copyMakeBorder(alpha, border, band + border, border, border,
                                       cv2.BORDER_CONSTANT, value=255)

    fh, fw = framed.shape[:2]
    font = cv2.FONT_HERSHEY_TRIPLEX
    scale = band / 55
    th = max(1, int(scale * 1.5))
    (tw, tht), _ = cv2.getTextSize(title, font, scale, th)
    if tw > fw * 0.7:
        scale *= fw * 0.7 / tw
        th = max(1, int(scale * 1.5))
        (tw, tht), _ = cv2.getTextSize(title, font, scale, th)
    ty = h + border + (band + tht) // 2
    cv2.putText(framed, title, (border * 2, ty), font, scale, accent_bgr, th, cv2.LINE_AA)

    if subtitle:
        s_scale = scale * 0.45
        (sw, sh), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, s_scale, 1)
        cv2.putText(framed, subtitle, (fw - sw - border * 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    s_scale, (90, 90, 90), max(1, th // 2), cv2.LINE_AA)

    # postage stamp in the top-right corner of the photo
    r = max(18, w // 16)
    cx, cy = fw - border - r - border, border + r + border
    cv2.circle(framed, (cx, cy), r, (250, 250, 250), -1, cv2.LINE_AA)
    cv2.circle(framed, (cx, cy), r, accent_bgr, max(2, r // 10), cv2.LINE_AA)
    cv2.circle(framed, (cx, cy), int(r * 0.75), accent_bgr, 1, cv2.LINE_AA)
    st_thick = max(1, r // 25)
    (aw, ah), _ = cv2.getTextSize(stamp, cv2.FONT_HERSHEY_SIMPLEX, 1.0, st_thick)
    st_scale = (r * 1.2) / aw  # make the word fit inside the inner ring
    (aw, ah), _ = cv2.getTextSize(stamp, cv2.FONT_HERSHEY_SIMPLEX, st_scale, st_thick)
    cv2.putText(framed, stamp, (cx - aw // 2, cy + ah // 2), cv2.FONT_HERSHEY_SIMPLEX,
                st_scale, accent_bgr, st_thick, cv2.LINE_AA)
    if new_alpha is not None:
        cv2.circle(new_alpha, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return framed, new_alpha


# ------------------------------------------------------------ display
def split_view(original, processed_bgr, position_pct):
    """Before/after: original on the left of the divider, result on the right."""
    h, w = processed_bgr.shape[:2]
    orig = cv2.resize(original, (w, h), interpolation=cv2.INTER_AREA)
    x = int(w * position_pct / 100)
    out = processed_bgr.copy()
    out[:, :x] = orig[:, :x]
    thick = max(2, w // 300)
    cv2.line(out, (x, 0), (x, h), (255, 255, 255), thick, cv2.LINE_AA)
    scale = w / 1400
    for label, lx in (("BEFORE", max(10, x - int(170 * scale) - 10)), ("AFTER", min(w - int(140 * scale), x + 15))):
        cv2.putText(out, label, (lx, int(45 * scale) + 10), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), thick * 3, cv2.LINE_AA)
        cv2.putText(out, label, (lx, int(45 * scale) + 10), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)
    return out


def checkerboard_preview(bgra, square=24):
    """Composite a BGRA image over a checkerboard so transparency is visible."""
    h, w = bgra.shape[:2]
    yy, xx = np.indices((h, w))
    board = (((yy // square + xx // square) % 2) * 60 + 170).astype(np.uint8)
    board = cv2.merge([board, board, board]).astype(np.float64)
    a = bgra[:, :, 3:4].astype(np.float64) / 255.0
    out = bgra[:, :, :3].astype(np.float64) * a + board * (1 - a)
    return out.astype(np.uint8)


def to_png_bytes(img):
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else None
