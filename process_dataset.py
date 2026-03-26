import os
import torch
from utils import process_image

input_folder = "dataset"
output_folder = "hands"

os.makedirs(output_folder, exist_ok=True)

for hand_class in os.listdir(input_folder):
    hand_class_load_path = os.path.join(input_folder, hand_class)
    hand_class_save_path = os.path.join(output_folder, hand_class)
    os.makedirs(hand_class_save_path, exist_ok=True)
    for img_name in os.listdir(hand_class_load_path):
        img_path = os.path.join(hand_class_load_path, img_name)
        img_tensor = process_image(img_path)
        if img_tensor is None:
            continue
        print(img_tensor.shape)
        save_path = os.path.join(hand_class_save_path, img_name.split('.')[0] + ".pth")
        torch.save(img_tensor, save_path)
print("Done")
