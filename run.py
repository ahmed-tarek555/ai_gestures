import cv2
import torch
from utils import process_image, model_path, recognize, save_gesture
from model import Model

model = Model()
model.load_state_dict(torch.load(model_path))
model.eval()

camera_source = 0

def detect(source):
    cap = cv2.VideoCapture(source)
    output_text = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = process_image(rgb)
        if img_tensor is None:
            continue
        embedding = model(img_tensor)
        result = recognize(embedding)
        output_text.append(result)
        cv2.putText(frame, f'{result}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "".join(output_text)

def register(source):
    cap = cv2.VideoCapture(source)
    embeddings_list = []
    name = input()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = process_image(rgb)
        if img_tensor is None:
            continue
        embedding = model(img_tensor)
        embeddings_list.append(embedding)

        cv2.imshow("Camera", frame)
        if len(embeddings_list) == 5:
            save_gesture(embeddings_list, name)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

register(camera_source)
