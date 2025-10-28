import os
import random
from PIL import Image, ImageDraw, ImageFont

# -----------------------
# Configuration
# -----------------------
OUTPUT_DIR = "data_examples/handwritten_test_A"
NUM_IMAGES = 30                     # Number of samples
IMG_SIZE = (96, 96)                 # Must match model input
TARGET_FILL_RATIO = 0.9             # How much of the image height the glyph should occupy (0.9 = 90%)
FONT_DIR = "fonts/handwritten_fonts"      # Folder containing handwriting fonts (.ttf or .otf)
TEXT = "A"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def measure_letter_size(font_path, font_size, text):
    """Render text to a temporary image to measure its bounding box."""
    font = ImageFont.truetype(font_path, font_size)
    dummy_img = Image.new("L", (IMG_SIZE[0] * 2, IMG_SIZE[1] * 2), 0)
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height


def find_optimal_font_size(font_path, text, target_height):
    """Binary search for a font size that makes the letter height ≈ target_height."""
    low, high = 10, 200  # reasonable font size range for 96x96
    best_size = 40
    for _ in range(10):  # iterate to converge
        mid = (low + high) // 2
        _, height = measure_letter_size(font_path, mid, text)
        if height < target_height:
            low = mid
            best_size = mid
        else:
            high = mid
    return best_size


def render_letter(font_path, text):
    """Render a single centered letter with consistent visual size."""
    # Determine optimal font size for consistent height
    target_height = IMG_SIZE[1] * TARGET_FILL_RATIO
    optimal_size = find_optimal_font_size(font_path, text, target_height)
    font = ImageFont.truetype(font_path, optimal_size)

    # Create canvas and draw centered text
    img = Image.new("RGB", IMG_SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Center positioning
    x = (IMG_SIZE[0] - text_w) // 2 - bbox[0]
    y = (IMG_SIZE[1] - text_h) // 2 - bbox[1]

    draw.text((x, y), text, font=font, fill=(0, 0, 0))
    return img


def generate_images():
    fonts = [
        os.path.join(FONT_DIR, f)
        for f in os.listdir(FONT_DIR)
        if f.lower().endswith((".ttf", ".otf"))
    ]
    if not fonts:
        raise FileNotFoundError(
            f"No handwriting fonts found in '{FONT_DIR}'. Please add some .ttf or .otf handwriting fonts."
        )

    print(f"✅ Found {len(fonts)} handwriting fonts. Generating {NUM_IMAGES} balanced, centered images...")

    for i in range(NUM_IMAGES):
        font_path = random.choice(fonts)
        img = render_letter(font_path, TEXT)
        filename = f"A_{i+1:03d}.jpg"
        img.save(os.path.join(OUTPUT_DIR, filename))
        print(f"🖋️  Saved: {filename}")

    print(f"\n🎉 Done! Generated {NUM_IMAGES} visually balanced handwritten-style '{TEXT}' images in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    generate_images()