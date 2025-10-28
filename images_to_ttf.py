import os
import subprocess
from pathlib import Path
import fontforge
import psMat

# -------- Configuration --------
IMAGES_DIR = "outputs/latin_batch"     # path to your generated images
OUTPUT_TTF = "outputs/generated_font.ttf"
FONT_NAME = "AI_LatinFont"
FONT_STYLE = "Regular"
CHARACTER_SET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

# Ensure output folder exists
os.makedirs(os.path.dirname(OUTPUT_TTF), exist_ok=True)

# Temporary folder for PBM/SVG conversions - use absolute path
TMP_DIR = Path("tmp_vectors").resolve()
TMP_DIR.mkdir(exist_ok=True)

def raster_to_svg(image_path: Path):
    """
    Convert raster image (jpg/png) -> SVG vector using potrace.
    Returns the path to the generated SVG file.
    """
    # Ensure tmp directory still exists
    TMP_DIR.mkdir(exist_ok=True)
    
    pbm_path = TMP_DIR / (image_path.stem + ".pbm")
    svg_path = TMP_DIR / (image_path.stem + ".svg")

    # Convert to black-white PBM using ImageMagick's convert
    # Normalize the image before conversion
    convert_cmd = [
        "convert", 
        str(image_path), 
        "-colorspace", "Gray",
        "-normalize",
        "-contrast-stretch", "2%x2%",
        "-threshold", "50%",
        "-morphology", "Close", "Disk:1",  # Fill small gaps
        str(pbm_path)
    ]
    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("WARNING: Failed to convert {} to PBM: {}".format(image_path, result.stderr.strip()))
        return None
    
    # Verify PBM was created
    if not pbm_path.exists():
        print("WARNING: PBM file was not created for {}".format(image_path))
        return None

    # Trace PBM -> SVG with optimized settings for consistent strokes
    potrace_cmd = [
        "potrace", 
        str(pbm_path), 
        "--svg", 
        "-o", str(svg_path),
        "--turdsize", "3",      # Remove small artifacts
        "--alphamax", "0.8",    # Corner smoothness
        "--opttolerance", "0.3" # Optimization tolerance
    ]
    result = subprocess.run(potrace_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("WARNING: Failed to trace {} to SVG: {}".format(pbm_path, result.stderr.strip()))
        return None
    
    # Verify SVG was created
    if not svg_path.exists():
        print("WARNING: SVG file was not created for {}".format(image_path))
        return None

    return svg_path


# -------- Create new font --------
font = fontforge.font()
font.encoding = "UnicodeFull"
font.fontname = FONT_NAME
font.fullname = FONT_NAME + " " + FONT_STYLE
font.familyname = FONT_NAME
font.em = 1000  # Units per em (standard)
font.ascent = 800  # Height above baseline
font.descent = 200  # Depth below baseline

added_count = 0

# Store glyph bounds for normalization
glyph_data = []

for char in CHARACTER_SET:
    image_path = Path(IMAGES_DIR) / "{}.jpg".format(char)
    if not image_path.exists():
        print("WARNING: Missing image for '{}', skipping.".format(char))
        continue

    svg_path = raster_to_svg(image_path)
    if not svg_path:
        print("WARNING: Could not vectorize {}".format(image_path))
        continue

    try:
        glyph = font.createChar(ord(char), char)
        glyph.importOutlines(str(svg_path))
        
        # Get bounding box before transformations
        bbox = glyph.boundingBox()
        glyph_data.append({
            'glyph': glyph,
            'char': char,
            'bbox': bbox
        })
        
        print("SUCCESS: Imported glyph: '{}' from {}".format(char, svg_path.name))
    except Exception as e:
        print("WARNING: Failed to import {}: {}".format(char, e))

if len(glyph_data) == 0:
    raise RuntimeError("No glyphs added. Aborting font generation.")

# -------- Normalize all glyphs --------
print("\nNormalizing glyphs...")

# Calculate target metrics based on typical proportions
TARGET_CAP_HEIGHT = 700  # Height for uppercase letters
TARGET_X_HEIGHT = 500    # Height for lowercase letters
BASELINE_Y = 0           # Standard baseline position

for data in glyph_data:
    glyph = data['glyph']
    char = data['char']
    bbox = data['bbox']
    
    # bbox format: (xmin, ymin, xmax, ymax)
    glyph_width = bbox[2] - bbox[0]
    glyph_height = bbox[3] - bbox[1]
    
    if glyph_width == 0 or glyph_height == 0:
        print("WARNING: Skipping '{}' - invalid dimensions".format(char))
        continue
    
    # Determine target height based on character type
    if char.isupper() or char.isdigit():
        target_height = TARGET_CAP_HEIGHT
    else:  # lowercase
        target_height = TARGET_X_HEIGHT
    
    # Calculate scale to fit target height
    scale_factor = target_height / glyph_height
    
    # Calculate proportional width
    scaled_width = glyph_width * scale_factor
    
    # Center horizontally with some padding
    left_bearing = 50  # Left side spacing
    right_bearing = 50  # Right side spacing
    
    # Clean up the glyph before transformation
    try:
        glyph.removeOverlap()
        glyph.correctDirection()
        glyph.simplify(1.0, ['cleanup'])
    except Exception as e:
        print("INFO: Pre-cleanup warning for '{}': {}".format(char, e))
    
    # Transform: scale and position
    # Step 1: Translate to origin
    glyph.transform(psMat.translate(-bbox[0], -bbox[1]))
    
    # Step 2: Scale uniformly (same scale for x and y to maintain aspect ratio)
    glyph.transform(psMat.scale(scale_factor, scale_factor))
    
    # Step 3: Position at baseline with left bearing
    glyph.transform(psMat.translate(left_bearing, BASELINE_Y))
    
    # Set character advance width (space to next character)
    glyph.width = int(scaled_width + left_bearing + right_bearing)
    
    # Final cleanup and correction
    try:
        glyph.simplify(0.5, ['cleanup', 'forcelines'])
        glyph.removeOverlap()
        glyph.correctDirection()
        glyph.round()
    except Exception as e:
        print("INFO: Post-cleanup warning for '{}': {}".format(char, e))
    
    added_count += 1
    print("  > Normalized '{}': width={}, height={}".format(char, glyph.width, int(target_height)))

print("\nNormalized {} glyphs".format(added_count))

# Set font-wide properties for better rendering
font.selection.all()
font.autoHint()

# Generate the font
font.generate(OUTPUT_TTF)
print("\nFont generated successfully -> {}".format(OUTPUT_TTF))
print("   - Total glyphs: {}".format(added_count))
print("   - Font family: {}".format(FONT_NAME))

# Clean up temporary files
print("\nCleaning up temporary files...")
import shutil
if TMP_DIR.exists():
    shutil.rmtree(TMP_DIR)
    print("Temporary directory removed.")