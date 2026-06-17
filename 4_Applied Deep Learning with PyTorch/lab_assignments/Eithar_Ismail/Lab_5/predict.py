from ultralytics import YOLO

# Load model
model = YOLO("SAIR_Courses/4_PyTorch/lab_assignments/Eithar_Ismail/best.pt")

# Predict
results = model.predict(
    source="SAIR_Courses/4_PyTorch/lab_assignments/Eithar_Ismail/Brain-With-Meningioma-Tumour.jpg",
    imgsz=640,
    conf=0.2,
    save=True
)

print("Prediction done!")

print(results)