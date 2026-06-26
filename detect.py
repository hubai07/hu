from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("./runs/train/exp/weights/best.pt")
    model.predict(
        source="./data/train/images/0b725f9f-frame_0740.jpg",
        imgsz=640,
        project='./datect',
        name='exp',
        save=True
    )
