from models.yolo_model import YOLOv9Model

def train_yolo():
    yolo = YOLOv9Model()
    yolo.train(data_path="datasets/yolo_data.yaml", epochs=50, batch_size=16, img_size=640)

if __name__ == "__main__":
    train_yolo()
