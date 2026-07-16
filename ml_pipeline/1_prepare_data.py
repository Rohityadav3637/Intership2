import os
import shutil
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

def rle_to_bbox(rle_str, height=256, width=1600):
    """
    Decodes column-major RLE format mask to bounding box coordinates (min_x, min_y, max_x, max_y).
    """
    if pd.isna(rle_str) or not isinstance(rle_str, str) or len(rle_str.strip()) == 0:
        return None
    try:
        s = np.fromstring(rle_str, dtype=int, sep=' ')
        starts = s[0::2] - 1
        lengths = s[1::2]
        
        # Reconstruct pixel indices
        indices = np.concatenate([np.arange(start, start + length) for start, length in zip(starts, lengths)])
        
        # Column-major coordinates conversion
        ys = indices % height
        xs = indices // height
        
        min_x = int(xs.min())
        max_x = int(xs.max())
        min_y = int(ys.min())
        max_y = int(ys.max())
        
        return min_x, min_y, max_x, max_y
    except Exception as e:
        print(f"Error decoding RLE: {e}")
        return None

def main():
    # Setup paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(base_dir, "data")
    train_images_src = os.path.join(data_dir, "train_images")
    csv_path = os.path.join(data_dir, "train.csv")
    
    # Check inputs
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"train.csv not found at {csv_path}")
    if not os.path.exists(train_images_src):
        raise FileNotFoundError(f"train_images directory not found at {train_images_src}")
        
    print("Reading train.csv...")
    df = pd.read_csv(csv_path)
    
    # Process column structures (Severstal CSV can be either ImageId_ClassId or separate columns)
    if "ImageId_ClassId" in df.columns:
        df["ImageId"] = df["ImageId_ClassId"].apply(lambda x: x.split("_")[0])
        df["ClassId"] = df["ImageId_ClassId"].apply(lambda x: int(x.split("_")[1]))
    
    # Filter rows with actual annotations
    df_annotations = df[df["EncodedPixels"].notna()].copy()
    
    # Create a mapping of ImageId -> list of bounding boxes in YOLO format
    # Each box: (class_idx, x_center, y_center, w, h)
    image_labels = {}
    print("Processing RLE annotations into YOLO bounding boxes...")
    for idx, row in tqdm(df_annotations.iterrows(), total=df_annotations.shape[0]):
        image_id = row["ImageId"]
        class_id = int(row["ClassId"])
        rle = row["EncodedPixels"]
        
        bbox = rle_to_bbox(rle)
        if bbox is not None:
            min_x, min_y, max_x, max_y = bbox
            
            # Convert to YOLO format (normalized center x, center y, width, height)
            dw = 1.0 / 1600.0
            dh = 1.0 / 256.0
            
            x_center = (min_x + max_x) / 2.0 * dw
            y_center = (min_y + max_y) / 2.0 * dh
            w = (max_x - min_x + 1.0) * dw
            h = (max_y - min_y + 1.0) * dh
            
            # Clamp to [0.0, 1.0] just in case
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)
            
            # YOLO classes are 0-indexed, ClassId is 1-indexed
            class_idx = class_id - 1
            
            if image_id not in image_labels:
                image_labels[image_id] = []
            image_labels[image_id].append(f"{class_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    # Find all images in the source directory
    all_images = [f for f in os.listdir(train_images_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(all_images)} total images in source directory.")
    
    # Shuffle and split (70/15/15)
    random.seed(42)
    random.shuffle(all_images)
    
    n_total = len(all_images)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_split = all_images[:n_train]
    val_split = all_images[n_train:n_train + n_val]
    test_split = all_images[n_train + n_val:]
    
    splits = {
        "train": train_split,
        "val": val_split,
        "test": test_split
    }
    
    # Prepare target directories
    for split in ["train", "val", "test"]:
        images_dest = os.path.join(data_dir, "images", split)
        labels_dest = os.path.join(data_dir, "labels", split)
        
        # Recreate directories to ensure fresh start
        if os.path.exists(images_dest):
            shutil.rmtree(images_dest)
        if os.path.exists(labels_dest):
            shutil.rmtree(labels_dest)
            
        os.makedirs(images_dest, exist_ok=True)
        os.makedirs(labels_dest, exist_ok=True)
        
    # Copy images and write labels
    print("Distributing files to splits...")
    split_counts = {"train": 0, "val": 0, "test": 0}
    for split_name, image_list in splits.items():
        images_dest = os.path.join(data_dir, "images", split_name)
        labels_dest = os.path.join(data_dir, "labels", split_name)
        
        for img_name in tqdm(image_list, desc=f"Copying {split_name} split"):
            # Copy image
            src_img_path = os.path.join(train_images_src, img_name)
            dest_img_path = os.path.join(images_dest, img_name)
            shutil.copy2(src_img_path, dest_img_path)
            
            # Write label file (empty if no defect)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            dest_label_path = os.path.join(labels_dest, label_name)
            
            boxes = image_labels.get(img_name, [])
            with open(dest_label_path, "w") as f:
                if boxes:
                    f.write("\n".join(boxes) + "\n")
            
            split_counts[split_name] += 1
            
    print("\nDataset preparation complete!")
    print(f"Train split: {split_counts['train']} images")
    print(f"Val split:   {split_counts['val']} images")
    print(f"Test split:  {split_counts['test']} images")
    
    # Create dataset.yaml
    yaml_content = f"""path: {data_dir.replace(os.sep, '/')}
train: images/train
val: images/val
test: images/test

names:
  0: defect_1
  1: defect_2
  2: defect_3
  3: defect_4
"""
    yaml_path = os.path.join(base_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Created dataset.yaml at {yaml_path}")

if __name__ == "__main__":
    main()
