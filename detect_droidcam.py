from ultralytics import YOLO
import cv2
from datetime import datetime

MODEL_PATH = "best.pt"
CAMERA_INDEX = 1
CONF_THRESHOLD = 0.7
IMG_SIZE = 640

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"[!] Kamera index {CAMERA_INDEX} tidak bisa dibuka.")
    exit()

print("[+] DroidCam aktif.")
print("[+] Tekan Q untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[!] Gagal membaca frame.")
        break

    results = model.predict(
        source=frame,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        verbose=False
    )

    result = results[0]
    annotated_frame = result.plot()

    deer_count = len(result.boxes)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        annotated_frame,
        f"{now} | Deer count: {deer_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("DroidCam Deer Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
