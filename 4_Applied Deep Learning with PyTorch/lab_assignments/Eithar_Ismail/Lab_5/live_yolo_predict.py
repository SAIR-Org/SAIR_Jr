import cv2
import time
import numpy as np
from ultralytics import YOLO


model = YOLO("SAIR_Courses/4_PyTorch/lab_assignments/Eithar_Ismail/Lab_5/best.pt")

def live_detection():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Webcam not available")
        return

    prev_time = time.time()

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        annotated = results.plot()

        # FPS
        fps = 1/(time.time() - prev_time)
        prev_time = time.time()

        cv2.putText(annotated,
                    f"FPS: {fps:.1f}",
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        cv2.imshow("Live Tumor Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_detection()