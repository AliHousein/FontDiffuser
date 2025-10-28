import os
import string
import math
import torch
from tqdm import tqdm
from PIL import Image

from sample import (
    arg_parse,
    load_fontdiffuer_pipeline,
    sampling,
    save_args_to_yaml
)

# -------- CONFIGURATION --------
CHARACTER_SET = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
SAVE_ROOT = "outputs/latin_batch"
os.makedirs(SAVE_ROOT, exist_ok=True)


def generate_preview_grid(image_folder, output_path="outputs/latin_batch/preview_grid.jpg",
                          thumb_size=(96, 96), cols=16, padding=10, bg_color=(255, 255, 255)):
    """Generate a preview grid combining all character images."""
    images = []
    for file in sorted(os.listdir(image_folder)):
        if file.lower().endswith((".png", ".jpg", ".jpeg")) and file not in ("preview_grid.jpg", "out_single.png"):
            try:
                img = Image.open(os.path.join(image_folder, file)).convert("RGB")
                img = img.resize(thumb_size, Image.LANCZOS)
                images.append((file, img))
            except Exception:
                pass

    if not images:
        print("⚠️ No images found to create preview grid.")
        return

    rows = math.ceil(len(images) / cols)
    grid_width = cols * thumb_size[0] + (cols + 1) * padding
    grid_height = rows * thumb_size[1] + (rows + 1) * padding

    grid = Image.new("RGB", (grid_width, grid_height), bg_color)

    x, y = padding, padding
    for idx, (_, img) in enumerate(images):
        grid.paste(img, (x, y))
        x += thumb_size[0] + padding
        if (idx + 1) % cols == 0:
            x = padding
            y += thumb_size[1] + padding

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grid.save(output_path)
    print(f"🖼️ Preview grid saved to: {output_path}")


def main():
    # Parse arguments
    args = arg_parse()

    # Force device to GPU if available
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fix save path
    args.save_image_dir = SAVE_ROOT

    # Ensure we control saving manually (avoid out_single.png)
    args.save_image = False

    # Save one shared sampling config
    config_path = os.path.join(SAVE_ROOT, "sampling_config.yaml")
    if not os.path.exists(config_path):
        save_args_to_yaml(args=args, output_file=config_path)

    # Load model once
    pipe = load_fontdiffuer_pipeline(args=args)
    print(f"✅ Model loaded on {args.device}. Ready for Latin batch generation.\n")

    # Disable saving of out_with_cs composite images
    import sample
    original_save_image_with_content_style = sample.save_image_with_content_style
    def skip_save_image_with_content_style(*_, **__): return
    sample.save_image_with_content_style = skip_save_image_with_content_style

    # Loop through all characters
    for char in tqdm(CHARACTER_SET, desc="Generating Latin Characters"):
        args.content_character = char
        filename = f"{char}.jpg" if char.isalnum() else f"char_{ord(char)}.jpg"
        output_path = os.path.join(SAVE_ROOT, filename)
        if os.path.exists(output_path):
            continue  # skip already generated images
        try:
            img = sampling(args=args, pipe=pipe)
            if img is not None:
                img.save(output_path)
        except Exception as e:
            print(f"⚠️ Failed to generate '{char}': {e}")

    # Restore original function
    sample.save_image_with_content_style = original_save_image_with_content_style

    # Create the preview grid
    print("\n🧩 Generating preview grid...")
    generate_preview_grid(image_folder=SAVE_ROOT)

    print(f"\n🎉 Batch generation complete! All characters saved in: {SAVE_ROOT}")


if __name__ == "__main__":
    main()