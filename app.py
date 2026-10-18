import hashlib
import os
import random

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import processing as p



@st.cache_resource(show_spinner=False)
def get_model(name):
    return p.load_model(name)


@st.cache_data(show_spinner=False, max_entries=8)
def colorized(image, name, strength):
    return p.colorize(image, name, get_model(name), strength)


def run_colorize(image, name, strength):
    """Colorize with one model; shows progress and errors in the page. None = failed."""
    if not p.model_available(name):
        lib = "onnxruntime" if name == "DDColor (tiny)" else "torch"
        st.warning(f"{name} needs the `{lib}` package. Run `py -m pip install -r requirements.txt`.")
        return None
    message = (f"Colorizing with {name}…" if p.model_downloaded(name)
               else f"Downloading {name} (first time only, about 130 MB)…")
    try:
        with st.spinner(message):
            return colorized(image, name, strength)
    except Exception as exc:
        st.error(f"{name} failed: {exc}")
        return None

st.set_page_config(page_title="Snap Remix", page_icon="📸", layout="wide")

st.title("📸 Snap Remix")
st.caption(
    "Remix any photo or PDF page: colorize black-and-white photos with AI, restyle them, swap the "
    "background with thresholding and alpha masks, add captions, make a postcard, and download the result. "
    "Built with OpenCV for Computer Vision, Module 2."
)


# Every control's key ends in a version number. "Reset" bumps the version, so all
# controls are recreated fresh with their default values.
st.session_state.setdefault("v", 0)


def K(name):
    return f"{name}_{st.session_state.v}"


# Defaults for controls that "Surprise me" can change
DEFAULTS = {"preset": "None", "remove_bg": False, "bg_choice": "Sunset sky",
            "postcard_on": False, "view": "Side by side"}


def apply_defaults():
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(K(k), v)


apply_defaults()


# ----------------------------------------------------------- buttons
def surprise():
    st.session_state[K("preset")] = random.choice(p.PRESETS[1:])
    st.session_state[K("remove_bg")] = True
    st.session_state[K("bg_choice")] = random.choice(["Sunset sky", "Ocean blue", "Starry night"])
    st.session_state[K("postcard_on")] = True
    st.session_state[K("view")] = "Before / after slider"


def reset():
    old = st.session_state.v
    for key in [k for k in st.session_state if str(k).endswith(f"_{old}")]:
        del st.session_state[key]
    st.session_state.v = old + 1
    apply_defaults()


# ------------------------------------------------------------------ input
with st.sidebar:
    b1, b2 = st.columns(2)
    b1.button("🎲 Surprise me", on_click=surprise)
    b2.button("↺ Reset", on_click=reset, help="Put every setting back to its default (keeps your photo).")

    st.header("1 · Image")
    upload = st.file_uploader("Upload a photo or PDF", type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "pdf"])
    use_sample = st.checkbox("Use a sample photo", value=upload is None)

if upload is not None and not use_sample:
    if upload.name.lower().endswith(".pdf"):
        try:
            data = upload.getvalue()
            img, pages = p.load_pdf_page(data, st.session_state.get("pdf_page", 1))
            if st.session_state.get("pdf_page", 1) > pages:  # a shorter PDF was uploaded
                st.session_state.pdf_page = 1
                img, pages = p.load_pdf_page(data, 1)
            if pages > 1:
                with st.sidebar:
                    st.number_input(f"PDF page (1–{pages})", 1, pages, key="pdf_page")
        except Exception:
            st.error("That PDF couldn't be opened. It may be damaged or password-protected.")
            st.stop()
    else:
        img = p.load_image(upload.getvalue())
        if img is None:
            st.error("That file couldn't be read as an image. Try a JPG, PNG or PDF.")
            st.stop()
elif use_sample:
    with open("sample.jpg", "rb") as f:
        img = p.load_image(f.read())
else:
    st.info("⬅️ Upload a photo or PDF in the sidebar, or tick **Use a sample photo**.")
    st.stop()

# --------------------------------------------------------------- controls
with st.sidebar:
    st.header("2 · Color & contrast")
    is_bw = p.looks_black_and_white(img)
    colorize_on = st.checkbox("Colorize with AI (for black-and-white photos)", key=K("colorize_on"),
                              help="Keeps the photo's lightness (L in Lab); a neural network predicts the two "
                                   "color channels (a, b).")
    if is_bw and not colorize_on:
        st.caption("📷 This photo looks black-and-white. Try **Colorize with AI**.")
    model_name = st.selectbox("Model", p.MODELS, index=2, disabled=not colorize_on, key=K("color_model"),
                              help="The three models from the homework notebook. Compare all three "
                                   "under the image.")
    if colorize_on:
        st.caption(p.MODEL_INFO[model_name])
    strength = st.slider("Color strength", 0.0, 2.0, 1.0, 0.1, disabled=not colorize_on,
                         help="Scales the predicted a and b channels: 0 = gray, 1 = as predicted, 2 = extra vivid.",
                         key=K("color_strength"))
    eq_mode = st.selectbox("Equalize histogram", p.EQUALIZE, key=K("equalize"),
                           help="Spreads brightness over 0–255 for more contrast. Equalizing each color "
                                "channel separately shifts the colors; equalize only V in HSV to keep them.")

    st.header("3 · Style")
    preset = st.selectbox("Look", p.PRESETS, key=K("preset"),
                          help="Golden Hour and Noir use image arithmetic, Pop Art thresholds each "
                               "color channel, Glitch overflows on purpose, and Scan uses adaptive "
                               "thresholding to clean up documents.")
    brightness = st.slider("Brightness (cv2.add / cv2.subtract)", -100, 100, 0, key=K("brightness_cv2_add_cv2_subtrac"))
    contrast = st.slider("Contrast (cv2.multiply)", 0.5, 2.0, 1.0, 0.05, key=K("contrast_cv2_multiply"))
    clip = st.checkbox("Clip to 0–255 (fixes overflow)", value=True,
                       help="Untick to see uint8 overflow: bright pixels wrap around to dark.", key=K("clip_to_0_255_fixes_overflow"))

    st.header("4 · Sky & background")
    remove_bg = st.checkbox("Cut out the background", key=K("remove_bg"))
    off = not remove_bg
    bg_choice = st.selectbox("Replace it with", p.BACKGROUNDS, key=K("bg_choice"), disabled=off)
    bg_color, bg_custom = (0, 0, 0), None
    if bg_choice == "Solid color" and not off:
        bg_color = p.hex_to_bgr(st.color_picker("Background color", "#1E90FF", key=K("background_color")))
    if bg_choice == "My own image" and not off:
        bg_file = st.file_uploader("Background image", type=["jpg", "jpeg", "png", "webp"], key=K("bg_file"))
        if bg_file is not None:
            bg_custom = p.load_image(bg_file.getvalue())

    method = st.selectbox("Threshold method", ["Global", "Otsu (automatic)", "Adaptive"], disabled=off, key=K("threshold_method"))
    keep_dark = st.checkbox("Keep the dark parts (THRESH_BINARY_INV)", value=True, disabled=off,
                            help="On: dark subjects (people, buildings, rocks) stay and the bright background is replaced. "
                                 "Off: bright subjects stay.", key=K("keep_the_dark_parts_thresh_bin"))
    thresh = st.slider("Threshold", 0, 255, 80, disabled=off or method != "Global", key=K("threshold"))
    block = st.slider("Adaptive block size (odd)", 3, 101, 31, 2, disabled=off or method != "Adaptive", key=K("adaptive_block_size_odd"))
    c_val = st.slider("Adaptive C", 0, 20, 7, disabled=off or method != "Adaptive", key=K("adaptive_c"))
    sky_only = st.checkbox("Only replace the top of the image", value=True, disabled=off,
                           help="Stops bright areas lower down (water, sand, a white table) from being replaced.", key=K("only_replace_the_top_of_the_ph"))
    sky_end = st.slider("Replace down to (% from top)", 5, 100, 50, disabled=off or not sky_only, key=K("sky_ends_at_from_top"))
    touch_top = st.checkbox("Background must touch the top edge", value=True, disabled=off,
                            help="Ignores bright spots inside the subject, like highlights or reflections.", key=K("sky_must_touch_the_top_edge"))
    feather_amt = st.slider("Soften edges (feather)", 0, 15, 2, disabled=off, key=K("soften_edges_feather"))
    bg_opacity = st.slider("Original-image opacity in the cut-out area", 0, 255, 0, disabled=off,
                           help="0 = fully replaced, 128 = half see-through (like the kangaroo), 255 = untouched.", key=K("original_photo_opacity_in_the_"))

    st.header("5 · Annotate")
    caption = st.text_input("Caption", "", key=K("caption"))
    cap_color = st.color_picker("Caption color", "#FFFFFF", key=K("caption_color"))
    cap_pos = st.radio("Caption position", ["Bottom", "Top"], horizontal=True, key=K("caption_position"))
    cap_size = st.slider("Caption size", 0.5, 3.0, 1.0, 0.1, key=K("caption_size"))
    cap_banner = st.checkbox("Black banner behind caption", value=True, key=K("black_banner_behind_caption"))

    add_box = st.checkbox("Highlight a region", key=K("highlight_a_region"))
    box_shape = st.radio("Shape", ["Rectangle", "Circle"], horizontal=True, disabled=not add_box, key=K("shape"))
    box_color = st.color_picker("Highlight color", "#FF00FF", disabled=not add_box, key=K("highlight_color"))
    box_x = st.slider("Left ↔ right (%)", 0, 100, (35, 65), disabled=not add_box, key=K("left_right"))
    box_y = st.slider("Top ↕ bottom (%)", 0, 100, (20, 50), disabled=not add_box, key=K("top_bottom"))

    st.header("6 · Postcard")
    postcard_on = st.checkbox("Turn it into a postcard", key=K("postcard_on"))
    pc_title = st.text_input("Postcard title", "Greetings!", disabled=not postcard_on, key=K("postcard_title"))
    pc_sub = st.text_input("Small print", "", disabled=not postcard_on, key=K("small_print"))
    pc_color = st.color_picker("Accent color", "#E67828", disabled=not postcard_on, key=K("accent_color"))
    pc_stamp = st.text_input("Stamp text", "SNAP", max_chars=8, disabled=not postcard_on, key=K("stamp_text"))

    st.header("7 · View")
    view = st.radio("Compare", ["Side by side", "Before / after slider"], key=K("view"))
    checker = st.checkbox("Show transparency as a checkerboard", value=True, key=K("show_transparency_as_a_checker"))

# --------------------------------------------------------------- pipeline
base = img
if colorize_on:
    colored = run_colorize(img, model_name, strength)
    if colored is not None:
        base = colored
base = p.equalize(base, eq_mode)
styled = p.apply_preset(base, preset)
adjusted = p.adjust(styled, brightness, contrast, clip)
result = adjusted.copy()
mask, used, alpha = None, None, None

if remove_bg:
    # The mask comes from the ORIGINAL photo so restyling doesn't change the cut-out
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask, used = p.make_mask(gray, method, thresh, keep_dark, block, c_val)
    alpha = p.build_alpha(mask, bg_opacity)
    if sky_only:
        alpha = p.limit_to_top(alpha, sky_end)
    if touch_top:
        alpha = p.sky_touching_top(alpha)
    alpha = p.feather(alpha, feather_amt)

    if bg_choice != "Transparent":
        h, w = result.shape[:2]
        bg = p.make_background(bg_choice, h, w, bg_color, bg_custom)
        if bg is None:
            st.sidebar.info("Upload a background image to use it.")
        else:
            result = p.composite(result, alpha, bg)
            alpha = None  # the new background fills it, so the result is solid again

if caption:
    p.draw_caption(result, alpha, caption, p.hex_to_bgr(cap_color), cap_pos, cap_size, cap_banner)
if add_box:
    p.draw_box(result, alpha, box_x, box_y, p.hex_to_bgr(box_color), box_shape)
if postcard_on:
    result, alpha = p.postcard(result, alpha, pc_title, pc_sub, p.hex_to_bgr(pc_color), pc_stamp.upper() or "SNAP")

final = p.attach_alpha(result, alpha) if alpha is not None else result


def for_display(image):
    """RGB(A) array for st.image."""
    if image.shape[2] == 4:
        if checker:
            return p.checkerboard_preview(image)[:, :, ::-1]
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return image[:, :, ::-1]


# ---------------------------------------------------------------- display
if view == "Side by side":
    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.image(img[:, :, ::-1])
    with right:
        st.subheader("Processed")
        st.image(for_display(final))
else:
    split = st.slider("Drag to compare", 0, 100, 50, label_visibility="collapsed", key=K("drag_to_compare"))
    shown = p.checkerboard_preview(final) if final.shape[2] == 4 else final
    st.image(p.split_view(img, shown, split)[:, :, ::-1])

if mask is not None:
    with st.expander("See the mask (white = kept, black = replaced)"):
        st.image(mask, clamp=True)
        if used is not None and method == "Otsu (automatic)":
            st.caption(f"Otsu picked a threshold of **{used:.0f}**.")
        kept = (mask > 0).mean() * 100
        st.caption(f"The threshold keeps {kept:.0f}% of the photo.")

with st.expander("🎨 Compare all three colorization models"):
    st.caption("Runs ECCV 2016, SIGGRAPH 2017 and DDColor on the original image and lines them up "
               "like the last cell of the homework notebook. Uses the Color strength slider.")
    img_key = (hashlib.md5(img.tobytes()).hexdigest(), strength)
    if st.button("Run all three models", key=K("compare_run")):
        results = [run_colorize(img, name, strength) for name in p.MODELS]
        if all(r is not None for r in results):
            grid = p.comparison_grid([img] + results, ["Original"] + p.MODELS)
            st.session_state["comparison"] = (img_key, p.to_png_bytes(grid))
    saved = st.session_state.get("comparison")
    if saved and saved[0] == img_key:
        st.image(saved[1])
        st.download_button("⬇️ Download the comparison", saved[1], file_name="comparison.png",
                           mime="image/png", key=K("compare_download"))

with st.expander("Histograms (cv2.calcHist)"):
    has_alpha = final.shape[2] == 4
    only_kept = st.checkbox("Count only the kept area (uses the alpha channel as a mask)",
                            disabled=not has_alpha, key=K("hist_only_kept"))
    names = ["Blue", "Green", "Red"]
    colors = ["#3b6fd8", "#2e9e4f", "#d64545"]
    h_left, h_right = st.columns(2)
    with h_left:
        st.markdown("**Original**")
        st.line_chart(pd.DataFrame(p.histograms(img), columns=names), color=colors, height=220)
    with h_right:
        st.markdown("**Processed**")
        kept = final[:, :, 3] if (has_alpha and only_kept) else None
        st.line_chart(pd.DataFrame(p.histograms(final[:, :, :3], kept), columns=names),
                      color=colors, height=220)
    st.caption("x = brightness 0–255 (left = shadows, right = highlights), y = number of pixels. "
               "A black-and-white photo has all three lines on top of each other; colorizing pulls them "
               "apart, and equalizing spreads them across the whole range.")

png = p.to_png_bytes(final)
d1, d2 = st.columns(2)
d1.download_button("⬇️ Download PNG", png, file_name="snap_remix.png", mime="image/png",
                   type="primary")
d2.download_button("⬇️ Download original + result side by side",
                   p.to_png_bytes(p.side_by_side(img, final)),
                   file_name="snap_remix_before_after.png", mime="image/png")

with st.expander("What's happening under the hood"):
    st.markdown(
        """
- **Image arithmetic**: brightness uses `cv2.add` / `cv2.subtract` with a constant matrix, and
  contrast uses `cv2.multiply`. With clipping off, values wrap past 255 (overflow). The *Golden Hour*
  look adds red and subtracts blue per channel; *Glitch* overflows on purpose.
- **AI colorization**: the photo goes to **Lab** color. **L** (lightness) is kept from the photo and a network
  predicts **a** (green↔red) and **b** (blue↔yellow) from a small copy of it: Zhang et al.'s **ECCV 2016** and
  **SIGGRAPH 2017** models (PyTorch, 256×256) or **DDColor** (tiny version, ONNX, 512×512). The predicted a, b
  are resized back up and joined with the full-size L, so the detail comes from the original photo.
- **Histograms & equalization**: `cv2.calcHist` counts pixels per brightness for each channel (optionally only
  inside the alpha mask). `cv2.equalizeHist` spreads them over 0–255; doing it on V in HSV keeps colors true,
  doing it on B, G and R separately shifts them.
- **PDFs**: each page is rendered to an image (pypdfium2) and then processed like any photo.
- **Thresholding**: global, Otsu (picks the threshold for you), or adaptive (a separate threshold per
  neighborhood). `THRESH_BINARY_INV` keeps dark areas. *Pop Art* thresholds each color channel separately,
  and *Scan* uses adaptive thresholding to turn a shadowy photo of a page into a clean black-and-white document.
- **Alpha channels & masks**: the mask is scaled to the chosen opacity, flipped with `cv2.bitwise_not`,
  softened with a Gaussian blur, and either attached as the 4th channel with `cv2.merge` (transparent PNG)
  or used to alpha-blend the photo over a new sky.
- **Drawing & annotation**: `cv2.putText` centered with `cv2.getTextSize`, filled `cv2.rectangle` banners,
  `cv2.rectangle` / `cv2.circle` highlights, the stars and sun in the skies, the postcard stamp and border
  (`cv2.copyMakeBorder`), and the `cv2.line` divider in the before/after view.
"""
    )
