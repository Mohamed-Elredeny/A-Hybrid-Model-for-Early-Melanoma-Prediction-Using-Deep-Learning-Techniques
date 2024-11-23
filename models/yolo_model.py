from ultralytics import YOLO

class YOLOv9Model:
    def __init__(self, pretrained_weights='yolov9.pt'):
        self.model = YOLO(pretrained_weights)

    def train(self, data_path, epochs, batch_size, img_size):
        self.model.train(
            data=data_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size
        )
        self.model.save('trained_yolov9_model.pt')
        print("YOLOv9 training complete.")

    def predict(self, images):
        return self.model.predict(source=images)

    def evaluate(self, data_path):
        return self.model.val(data=data_path)
