import os
import cv2
import logging
import shutil
from pathlib import Path
from models import SCRFD

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("dataset_cleaning.log")
        ]
    )

def clean_face_dataset(
    faces_dir: str = "face_dataset_small+faces-all",
    failed_dir: str = "failed_detections",
    det_weight: str = "./weights/det_10g.onnx"
):
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create detector
    detector = SCRFD(det_weight, input_size=(640, 640))
    
    # Create directory for failed detections
    os.makedirs(failed_dir, exist_ok=True)
    
    # Statistics
    total_images = 0
    failed_images = 0
    
    # Process each image
    for filename in os.listdir(faces_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        total_images += 1
        image_path = os.path.join(faces_dir, filename)
        
        # Read and detect faces
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not read image: {filename}")
            continue
            
        bboxes, kpss = detector.detect(image, max_num=1)
        
        # If no face detected, move to failed directory
        if len(kpss) == 0:
            failed_images += 1
            shutil.move(image_path, os.path.join(failed_dir, filename))
            logger.warning(f"No face detected in {filename}. Moved to {failed_dir}")
    
    # Log summary
    logger.info(f"Dataset cleaning completed:")
    logger.info(f"Total images processed: {total_images}")
    logger.info(f"Images with failed detection: {failed_images}")
    logger.info(f"Success rate: {((total_images-failed_images)/total_images)*100:.2f}%")

if __name__ == "__main__":
    clean_face_dataset()
