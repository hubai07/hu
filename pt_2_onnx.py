from ultralytics import YOLO
model = YOLO('D:/Computer_Vision/ultralytics-main/runs/train/exp/weights/best.pt')
model.export(format = 'onnx')