import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn.functional as F

model_path = "models/parameters.pth"

batch_size = 64
input_size = 63
classes = ['closedFist', 'fingerCircle', 'fingerSymbols', 'multiFingerBend', 'openPalm', 'semiOpenFist', 'semiOpenPalm', 'singleFingerBend']

base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

def process_image(image):
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image
    )
    results = detector.detect(image)
    if len(results.hand_landmarks) == 0:
        print("No hands detected")
        return None
    hand = results.hand_landmarks[0]
    landmarks = [(lm.x, lm.y, lm.z) for lm in hand]
    landmarks = torch.tensor(landmarks, dtype=torch.float32)
    output = landmarks.view(-1)
    return output

def process_dataset(input_folder, output_folder):
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

def load_dataset(dataset_path):
    x, y, classes = [], [], []
    idx = 0
    for class_name in os.listdir(dataset_path):
        classes.append(class_name)
        class_path = os.path.join(dataset_path, class_name)
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            tensor = torch.load(file_path)
            x.append(tensor)
            y.append(idx)
        idx += 1
    x = torch.stack(x, dim=0)
    y = torch.tensor(y)
    return x, y, classes

def get_batch(x, y):
    batch = torch.randint(x.shape[0], (batch_size, ))
    xb = x[batch]
    yb = y[batch]
    return xb, yb

def save_gesture(embedding_list, target):
    embedding = torch.stack(embedding_list, dim=0).mean(dim=0, keepdim=False)
    os.makedirs(f"alphabet/{target}", exist_ok=True)
    torch.save(embedding, f"alphabet/{target}/{target}.pth")

def load_known_embeddings(dir):
    known_embddings = {}
    for char_name in os.listdir(dir):
        char_path = os.path.join(dir, char_name)
        embedding = os.listdir(char_path)[0]
        known_embddings[char_name] = embedding
    return known_embddings

def recognize(input_embedding, threshold = 1.25):
    known_embeddings = load_known_embeddings("alphabet")
    best_distance = float('inf')
    result = None
    for name, embedding in known_embeddings.items():
        distance = torch.norm(input_embedding - embedding)
        if distance < best_distance:
            best_distance = distance
            result = name
    if best_distance > threshold:
        return None
    return result

