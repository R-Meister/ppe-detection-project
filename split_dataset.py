import os
import random
import shutil

# Set seed for reproducibility
random.seed(42)

# Directories
dataset_dir = 'datasets'
original_images_dir = os.path.join(dataset_dir, 'train/images')
original_labels_dir = os.path.join(dataset_dir, 'train/labels')

# Target folders
sets = ['train', 'valid', 'test']
for s in sets:
    os.makedirs(os.path.join(dataset_dir, s, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, s, 'labels'), exist_ok=True)

# Gather all image files
image_files = [f for f in os.listdir(original_images_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(image_files)

# Ratios
total = len(image_files)
train_split = int(0.7 * total)
valid_split = int(0.2 * total)

train_files = image_files[:train_split]
valid_files = image_files[train_split:train_split + valid_split]
test_files = image_files[train_split + valid_split:]

def move_files(file_list, split):
    for img_name in file_list:
        label_name = os.path.splitext(img_name)[0] + '.txt'

        img_src = os.path.join(original_images_dir, img_name)
        label_src = os.path.join(original_labels_dir, label_name)

        img_dst = os.path.join(dataset_dir, split, 'images', img_name)
        label_dst = os.path.join(dataset_dir, split, 'labels', label_name)

        shutil.move(img_src, img_dst)
        if os.path.exists(label_src):
            shutil.move(label_src, label_dst)
        else:
            print(f"[!] Warning: Label not found for {img_name}")

# Move files
move_files(train_files, 'train')
move_files(valid_files, 'valid')
move_files(test_files, 'test')

print(f"✅ Split complete: {len(train_files)} train, {len(valid_files)} valid, {len(test_files)} test")
