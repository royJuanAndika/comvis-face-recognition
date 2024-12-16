import cv2
import numpy as np
import argparse
from models import SCRFD
import os
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Face Detection and Keypoints")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--det-weight",
        type=str,
        default="./weights/det_10g.onnx",
        help="Path to detection model"
    )
    return parser.parse_args()

def draw_detection(image, bbox, keypoints, color=(0, 255, 0)):
    # Draw bounding box
    x1, y1, x2, y2, _ = bbox.astype(np.int32)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Keypoint colors - right eye, left eye, nose, right mouth, left mouth
    kp_colors = [
        (255, 0, 0),   # Right eye - Blue
        (0, 0, 255),   # Left eye - Red
        (0, 255, 0),   # Nose - Green
        (255, 0, 255), # Right mouth - Magenta
        (255, 255, 0)  # Left mouth - Cyan
    ]
    
    # Draw keypoints with numbers
    for i, (kp, kp_color) in enumerate(zip(keypoints, kp_colors)):
        x, y = kp.astype(np.int32)
        # Draw larger circle
        cv2.circle(image, (x, y), 4, kp_color, -1)  # Filled circle
        cv2.circle(image, (x, y), 5, (255, 255, 255), 1)  # White border
        
        # Add keypoint number
        cv2.putText(image, str(i), (x+5, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, kp_color, 2)
    
    # Add legend
    legend_text = ["0: Right Eye", "1: Left Eye", "2: Nose", 
                  "3: Right Mouth", "4: Left Mouth"]
    for i, text in enumerate(legend_text):
        cv2.putText(image, text, (10, 30 + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, kp_colors[i], 2)
        
    return image

def main():
    args = parse_args()
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Could not read image: {args.image}")
    
    # Initialize detector
    detector = SCRFD(args.det_weight, input_size=(640, 640))
    
    # Detect faces
    bboxes, kpss = detector.detect(image)
    
    # Draw results
    for bbox, kps in zip(bboxes, kpss):
        image = draw_detection(image, bbox, kps)
    
    # Create output directory
    output_dir = "./visualize_detection_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(args.image)
    filename = f"detection_{timestamp}_{base_name}"
    output_path = os.path.join(output_dir, filename)
    
    # Save the result
    cv2.imwrite(output_path, image)
    print(f"Found {len(bboxes)} faces in the image")
    print(f"Saved visualization to: {output_path}")
    
    # Show results
    cv2.namedWindow("Detection Results", cv2.WINDOW_NORMAL)
    cv2.imshow("Detection Results", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
