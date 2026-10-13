# 📸 Snap Remix

A Streamlit + OpenCV web app for Computer Vision, Module 2.

Upload any photo, or a PDF page, and remix it: restyle it, swap the background, add captions, turn it into a postcard, and download the result. It uses all four required concepts:

1. **Image arithmetic**: brightness with `cv2.add` / `cv2.subtract`, contrast with `cv2.multiply`, a clipping toggle that shows uint8 overflow, and one-click looks (*Golden Hour* shifts color channels, *Noir* boosts grayscale contrast, *Glitch* overflows on purpose).
2. **Thresholding**: global, Otsu or adaptive, with `THRESH_BINARY_INV` to keep dark subjects. *Pop Art* thresholds each color channel separately, and *Scan* uses adaptive thresholding to turn a shadowy photo of a page into a clean black-and-white document.
3. **Alpha channels and masks**: the mask becomes an alpha channel (`bitwise_not` + `merge`). It can be limited to the top of the image and to regions touching the top edge, and feathered with a Gaussian blur. It's then either saved as a transparent PNG or used to alpha-blend the image over a new background (sunset, ocean, starry night, solid color, or a second image).
4. **Drawing and annotation**: centered captions on a banner, rectangle or circle highlights, drawn stars and sun, a postcard border (`copyMakeBorder`) with a custom stamp, and a before/after view with a `cv2.line` divider.

Extras: PDF support (pick any page), a 🎲 **Surprise me** button, a ↺ **Reset** button, a before/after slider, a mask viewer, and a PNG download. A sample photo is included so the app works without an upload.

## Files

| File | What it is |
|---|---|
| `app.py` | The Streamlit page: upload, sidebar controls, display, download |
| `processing.py` | All the OpenCV functions |
| `requirements.txt` | Packages to install |
| `sample.jpg` | Sample photo |

## Run locally

Open Command Prompt in this folder (type `cmd` in File Explorer's address bar), then:

```
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a new **public** repository on GitHub, for example `snap-remix`.
2. Upload all four files to the top level of the repo (**Add file → Upload files**), then commit.
3. Go to https://share.streamlit.io and sign in with GitHub.
4. Click **Create app → Deploy a public app from GitHub**.
5. Pick your repo, branch `main`, main file `app.py`, and click **Deploy**.
6. You get a public URL like `https://snap-remix.streamlit.app` that anyone can open without logging in.
