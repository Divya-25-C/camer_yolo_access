import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Initialize classifier and detector
classifier = MobileNetV2(weights="imagenet")
detector = YOLO("yolov8n.pt")

# Open camera
camera = cv2.VideoCapture(0)
mode = "mobilenet"  # default mode

while camera.isOpened():
    ok, frame = camera.read()
    if not ok:
        break

    if mode == "yolo":
        results = detector(frame, verbose=False)
        frame = results[0].plot()
    else:
        # Preprocess for MobileNetV2
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_input(np.expand_dims(img.astype(np.float32), axis=0))

        preds = classifier.predict(img, verbose=0)
        decoded = decode_predictions(preds, top=1)[0][0]
        name, score = decoded[1], decoded[2]

        cv2.putText(frame, f"{name} {score:.0%}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("m=MobileNet, y=YOLO, q=quit", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    mode = {"m": "mobilenet", "y": "yolo"}.get(chr(key), mode)

camera.release()
cv2.destroyAllWindows()
