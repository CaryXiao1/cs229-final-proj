"""
tint_images.py
-------------------
This python script saves a randomly tinted copy of every image in
a folder, including all its subfolders. After running `python3 tint_images.py`,
you can specify the folder to duplicate photos. Within each sub-directory,
a folder named 'shifted' is created and all copies are saved there.
"""

import os
import random
from PIL import Image, ImageOps

def tint_image(image_path, output_path):
    """
    Apply a tint to the image by randomly reducing the red, green, and blue channels,
    ensuring pixel values remain valid, and preserving orientation.
    """
    red_tint = random.randint(0, 50)
    green_tint = random.randint(0, 50)
    blue_tint = random.randint(0, 50)

    with Image.open(image_path) as img:
        # preserve original orientation
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        pixels = img.load()

        for y in range(img.height):
            for x in range(img.width):
                r, g, b = pixels[x, y]
                r = max(0, r - red_tint)
                g = max(0, g - green_tint)
                b = max(0, b - blue_tint)
                pixels[x, y] = (r, g, b)

        img.save(output_path)

def process_directory(directory):
    """
    Traverse all subdirectories, find .jpeg images, and create tinted copies with '_red' added to filenames.
    Skip processing images in 'tinted_copies' folders.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".jpeg"):
                input_path = os.path.join(root, file)
                output_dir = os.path.join(root, "shifted")
                os.makedirs(output_dir, exist_ok=True)
                
                # add '_red' to the filename
                base_name, ext = os.path.splitext(file)
                output_file = f"{base_name}_red{ext}"
                output_path = os.path.join(output_dir, output_file)
                
                print(f"Processing {input_path} -> {output_path}")
                tint_image(input_path, output_path)

target_directory = input("Enter the path to the directory to process: ").strip()
process_directory(target_directory)
