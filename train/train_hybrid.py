from models.yolo_model import YOLOv9Model
from models.faster_rcnn_model import FasterRCNNModel
from models.hybrid_model import MelanoHybridModel
from torch.optim import Adam
from dataset import MelanomaDataset, transform
from torch.utils.data import DataLoader

def train_hybrid():
    yolo = YOLOv9Model()
    rcnn = FasterRCNNModel()
    hybrid_model = MelanoHybridModel(yolo, rcnn)

    train_dataset = MelanomaDataset("datasets/hybrid_data.json", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    optimizer = Adam(hybrid_model.parameters(), lr=0.001)

    hybrid_model.train()
    for epoch in range(50):
        for images, targets in train_loader:
            optimizer.zero_grad()
            predictions = hybrid_model(images)
            loss = torch.tensor(0.0)  # Define custom loss here
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} complete.")

    torch.save(hybrid_model.state_dict(), "trained_hybrid_model.pth")
    print("Hybrid model training complete.")

if __name__ == "__main__":
    train_hybrid()
