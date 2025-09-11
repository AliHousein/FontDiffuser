import os
import random
import string
import shutil
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps


# --------------------------
# CONFIG
# --------------------------
CANVAS_SIZE = 128
MARGIN = 10
VAL_RATIO = 0.2
CHARSET = string.ascii_uppercase + string.ascii_lowercase + string.digits + ".,!?;:'\"()-_@#&%"

# Use the already-downloaded skeleton font
CONTENT_FONT_PATH = Path("fonts/NotoSans-Italic-VariableFont_wdth,wght.ttf")

# Destination dirs
ROOT = Path("data_examples")
FONTS_DIR = Path("fonts/google_fonts")
FONTS_DIR.mkdir(parents=True, exist_ok=True)

# Read API key from environment variable
GOOGLE_FONTS_API_KEY = os.environ.get("GOOGLE_FONTS_API_KEY")
if not GOOGLE_FONTS_API_KEY:
    raise RuntimeError("Please set the GOOGLE_FONTS_API_KEY environment variable.")

GOOGLE_FONTS_API = f"https://www.googleapis.com/webfonts/v1/webfonts?key={GOOGLE_FONTS_API_KEY}&sort=popularity"


# --------------------------
# HELPERS
# --------------------------
def download_file(url, path):
    if not path.exists():
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"Downloaded {path.name}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")


def render_char(ch, font_path):
    """Render a character centered and normalized on a square canvas."""
    # Start with large font size, reduce dynamically if needed
    font_size = CANVAS_SIZE - 2 * MARGIN
    while font_size > 10:  # avoid infinite loop
        font = ImageFont.truetype(font_path, font_size)
        img_tmp = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        draw = ImageDraw.Draw(img_tmp)

        # Draw at (0,0) just to measure
        bbox = draw.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        if w <= CANVAS_SIZE - 2 * MARGIN and h <= CANVAS_SIZE - 2 * MARGIN:
            break  # fits
        font_size -= 2

    # Render final image
    img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), ch, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Center
    x = (CANVAS_SIZE - w) // 2 - bbox[0]
    y = (CANVAS_SIZE - h) // 2 - bbox[1]
    draw.text((x, y), ch, font=font, fill=0)

    # Binarize → remove gray artifacts
    img = img.point(lambda x: 0 if x < 200 else 255, mode="1").convert("L")

    # Normalize margins
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
        img = ImageOps.pad(img, (CANVAS_SIZE, CANVAS_SIZE), color=255, centering=(0.5, 0.5))

    return img


def prepare_fonts(limit=100):
    """Use local content font + download a batch of Google Fonts."""
    # No download for content font, just use the local path
    print("Fetching font list from Google Fonts API...")
    resp = requests.get(GOOGLE_FONTS_API, timeout=60)
    resp.raise_for_status()
    fonts_data = resp.json()["items"]

    random.shuffle(fonts_data)
    selected = fonts_data[:limit]

    font_paths = []
    for idx, item in enumerate(selected):
        url = item["files"].get("regular")
        if not url:
            continue
        out_path = FONTS_DIR / f"style{idx}.ttf"
        download_file(url, out_path)
        if out_path.exists():
            font_paths.append((f"style{idx}", out_path))

    return CONTENT_FONT_PATH, font_paths


def build_dataset(content_font_path, style_fonts):
    # Clear old dataset
    if ROOT.exists():
        shutil.rmtree(ROOT)

    content_dir = ROOT / "train" / "ContentImage"
    target_root = ROOT / "train" / "TargetImage"
    content_dir.mkdir(parents=True, exist_ok=True)
    target_root.mkdir(parents=True, exist_ok=True)

    # Generate shared content images
    print("Generating ContentImage set...")
    for idx, ch in enumerate(CHARSET):
        try:
            img = render_char(ch, str(content_font_path))
            img.save(content_dir / f"char{idx}.png")
        except Exception as e:
            print(f"  Skipped {ch} in content: {e}")

    # Generate TargetImage per style
    print("Generating TargetImage sets...")
    for style_name, font_path in style_fonts:
        style_dir = target_root / style_name
        style_dir.mkdir(parents=True)
        for idx, ch in enumerate(CHARSET):
            try:
                img = render_char(ch, str(font_path))
                img.save(style_dir / f"{style_name}+char{idx}.png")
            except Exception as e:
                print(f"  Skipped {ch} in {style_name}: {e}")


if __name__ == "__main__":
    content_font_path, style_fonts = prepare_fonts(limit=200)  # download 200 fonts by default
    build_dataset(content_font_path, style_fonts)
    print("✅ Dataset ready under ./data_examples/train/{ContentImage,TargetImage}/")