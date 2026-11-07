from ultralytics import YOLO

model = YOLO("best.pt") # Zubair se file lein

def predict_image(image_path):
    results = model(image_path)
    defects = []
    for r in results:
        for box in r.boxes:
            defects.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf)
            })
    return defects
