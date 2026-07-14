import os
import random
import cv2

def draw_yolo_boxes(image_path, label_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image: {image_path}")
        return
    
    h, w = img.shape[:2]
    
    # Color palette (BGF/RGB structure for OpenCV: BGR)
    colors = [
        (0, 255, 0),    # Green - Class 0 (Defect 1)
        (0, 0, 255),    # Red - Class 1 (Defect 2)
        (255, 0, 0),    # Blue - Class 2 (Defect 3)
        (0, 255, 255)   # Yellow - Class 3 (Defect 4)
    ]
    
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            box_w = float(parts[3]) * w
            box_h = float(parts[4]) * h
            
            x1 = int(x_center - box_w / 2.0)
            y1 = int(y_center - box_h / 2.0)
            x2 = int(x_center + box_w / 2.0)
            y2 = int(y_center + box_h / 2.0)
            
            # Clip coordinates to image boundary
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            
            # Select class-specific color
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background + text
            label_text = f"Class {class_id}"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(img, label_text, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
    # Save output as PNG
    cv2.imwrite(output_path, img)

def main():
    base_dir = r"c:\Users\Rishuraj Kumar\steelsight"
    train_image_dir = os.path.join(base_dir, "data", "images", "train")
    train_label_dir = os.path.join(base_dir, "data", "labels", "train")
    output_dir = os.path.join(base_dir, "notebooks", "verification_output")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all training images
    images = [f for f in os.listdir(train_image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print(f"No images found in {train_image_dir}")
        return
        
    # Pick from images with non-empty label files (defect present) to verify visually
    labeled_images = []
    for img in images:
        lbl_name = os.path.splitext(img)[0] + ".txt"
        lbl_path = os.path.join(train_label_dir, lbl_name)
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            labeled_images.append(img)
            
    print(f"Found {len(labeled_images)} train images with defect annotations.")
    
    # Sample 5 images
    if len(labeled_images) >= 5:
        selected_images = random.sample(labeled_images, 5)
    else:
        selected_images = random.sample(images, min(len(images), 5))
        
    print(f"Selected images for verification: {selected_images}")
    
    for img_name in selected_images:
        img_path = os.path.join(train_image_dir, img_name)
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(train_label_dir, lbl_name)
        
        # Save as PNG
        out_name = os.path.splitext(img_name)[0] + "_verified.png"
        out_path = os.path.join(output_dir, out_name)
        
        draw_yolo_boxes(img_path, lbl_path, out_path)
        print(f"Saved verified visualization to: {out_path}")
        
if __name__ == "__main__":
    main()
