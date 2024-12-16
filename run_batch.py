import os
import subprocess
from pathlib import Path

def process_test_images(test_dir='./test_data'):
    # Supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png')
    
    # Create test_dir if it doesn't exist
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all image files
    image_files = [
        f for f in Path(test_dir).iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    # Process each image
    for img_path in image_files:
        print(f"Processing: {img_path}")
        subprocess.run([
            'python', 
            'main-photo.py',
            '--source', 
            str(img_path)
        ])

if __name__ == "__main__":
    process_test_images()
