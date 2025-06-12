import os
import random
import shutil

# Set seed for reproducibility
random.seed(42)

# Directories
dataset_dir = 'datasets'
train_images_dir = os.path.join(dataset_dir, 'train/images')
train_labels_dir = os.path.join(dataset_dir, 'train/labels')
valid_images_dir = os.path.join(dataset_dir, 'valid/images')
valid_labels_dir = os.path.join(dataset_dir, 'valid/labels')

# Create valid folders if not exist
os.makedirs(valid_images_dir, exist_ok=True)
os.makedirs(valid_labels_dir, exist_ok=True)

# Get all image filenames
image_files = [f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.png'))]

# Shuffle and split
split_ratio = 0.2  # 20% for validation
val_count = int(len(image_files) * split_ratio)
val_images = random.sample(image_files, val_count)

print(f"Total images: {len(image_files)}")
print(f"Moving {val_count} images to validation set...")

# Move selected images and their labels
for img_name in val_images:
    label_name = os.path.splitext(img_name)[0] + '.txt'

    # Source paths
    img_src = os.path.join(train_images_dir, img_name)
    label_src = os.path.join(train_labels_dir, label_name)

    # Target paths
    img_dst = os.path.join(valid_images_dir, img_name)
    label_dst = os.path.join(valid_labels_dir, label_name)

    # Move files
    shutil.move(img_src, img_dst)
    if os.path.exists(label_src):
        shutil.move(label_src, label_dst)
    else:
        print(f"Warning: Label not found for {img_name}")

print("✅ Splitting complete.")
