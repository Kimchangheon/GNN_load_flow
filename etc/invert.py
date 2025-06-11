import os
from PIL import Image, ImageOps

# Input and output directories
input_dir = 'images_invert'
output_dir = 'inverted_output'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(image_extensions):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Open the image
        with Image.open(input_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Invert the image
            inverted_img = ImageOps.invert(img)

            # Save the inverted image
            inverted_img.save(output_path)
            print(f"Inverted: {filename} → {output_path}")