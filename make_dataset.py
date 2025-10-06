import os
from dotenv import load_dotenv
load_dotenv()
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
CONTENT_FONT_PATH = Path("fonts/NotoSans-VariableFont_wdth,wght.ttf")

# Destination dirs
ROOT = Path("data_examples")
FONTS_DIR = Path("fonts/google_fonts")
FONTS_DIR.mkdir(parents=True, exist_ok=True)

# Read API key
GOOGLE_FONTS_API_KEY = os.environ.get("GOOGLE_FONTS_API_KEY")
if not GOOGLE_FONTS_API_KEY:
    raise RuntimeError("Please set the GOOGLE_FONTS_API_KEY environment variable.")

GOOGLE_FONTS_API = f"https://www.googleapis.com/webfonts/v1/webfonts?key={GOOGLE_FONTS_API_KEY}&sort=popularity"

# --------------------------
# HELPERS
# --------------------------
def download_file(url, path):
    """Download a file from URL safely."""
    if path.exists():
        return
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"✅ Downloaded {path.name}")
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")


def render_char(ch, font_path):
    """Render a character centered and normalized on a square canvas."""
    try:
        font_size = CANVAS_SIZE - 2 * MARGIN
        while font_size > 10:
            font = ImageFont.truetype(font_path, font_size)
            img_tmp = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
            draw = ImageDraw.Draw(img_tmp)
            bbox = draw.textbbox((0, 0), ch, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if w <= CANVAS_SIZE - 2 * MARGIN and h <= CANVAS_SIZE - 2 * MARGIN:
                break
            font_size -= 2

        img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (CANVAS_SIZE - w) // 2 - bbox[0]
        y = (CANVAS_SIZE - h) // 2 - bbox[1]
        draw.text((x, y), ch, font=font, fill=0)
        img = img.point(lambda x: 0 if x < 200 else 255, mode="1").convert("L")
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
            img = ImageOps.pad(img, (CANVAS_SIZE, CANVAS_SIZE), color=255, centering=(0.5, 0.5))
        return img
    except Exception:
        return None


def validate_font(font_path):
    """Check if a font can render a few basic English characters."""
    try:
        font = ImageFont.truetype(font_path, 64)
        test_chars = ["A", "B", "a", "b", "1", "?"]
        for ch in test_chars:
            _ = font.getmask(ch)
        return True
    except Exception:
        return False


def prepare_fonts(max_fonts=None):
    """Fetch English (Latin) fonts only, download and validate them."""
    print("🔍 Fetching font list from Google Fonts API...")
    resp = requests.get(GOOGLE_FONTS_API, timeout=120)
    resp.raise_for_status()
    fonts_data = resp.json()["items"]

    # Filter fonts that support English (latin subset)
    english_fonts = [f for f in fonts_data if "latin" in f.get("subsets", [])]
    print(f"✅ Found {len(english_fonts)} English fonts on Google Fonts.")

    # Shuffle and limit
    random.shuffle(english_fonts)
    if max_fonts:
        english_fonts = english_fonts[:max_fonts]

    font_paths = []
    for idx, item in enumerate(english_fonts):
        url = item["files"].get("regular")
        if not url:
            continue
        out_path = FONTS_DIR / f"style{idx}.ttf"
        download_file(url, out_path)

        if not out_path.exists():
            continue

        # Validate font (skip broken ones)
        if validate_font(out_path):
            font_paths.append((f"style{idx}", out_path))
        else:
            print(f"⚠️ Skipping invalid or corrupted font: {out_path.name}")

    print(f"✅ {len(font_paths)} valid English fonts ready.")
    return CONTENT_FONT_PATH, font_paths


def build_dataset(content_font_path, style_fonts):
    """Render characters for content and style images."""
    if ROOT.exists():
        shutil.rmtree(ROOT)

    content_dir = ROOT / "train" / "ContentImage"
    target_root = ROOT / "train" / "TargetImage"
    content_dir.mkdir(parents=True, exist_ok=True)
    target_root.mkdir(parents=True, exist_ok=True)

    print("🖋️ Generating ContentImage set...")
    for idx, ch in enumerate(CHARSET):
        try:
            img = render_char(ch, str(content_font_path))
            if img:
                img.save(content_dir / f"char{idx}.jpg")
        except Exception as e:
            print(f"  ⚠️ Skipped {ch} in content: {e}")

    print("🎨 Generating TargetImage sets...")
    for style_name, font_path in style_fonts:
        style_dir = target_root / style_name
        style_dir.mkdir(parents=True)
        for idx, ch in enumerate(CHARSET):
            try:
                img = render_char(ch, str(font_path))
                if img:
                    img.save(style_dir / f"{style_name}+char{idx}.jpg")
            except Exception as e:
                print(f"  ⚠️ Skipped {ch} in {style_name}: {e}")

    print("✅ Dataset successfully built under ./data_examples/train/{ContentImage, TargetImage}/")


if __name__ == "__main__":
    # Download ALL English fonts (or set limit if needed)
    content_font_path, style_fonts = prepare_fonts(max_fonts=None)
    build_dataset(content_font_path, style_fonts)
    print("🎉 Dataset creation completed successfully!")