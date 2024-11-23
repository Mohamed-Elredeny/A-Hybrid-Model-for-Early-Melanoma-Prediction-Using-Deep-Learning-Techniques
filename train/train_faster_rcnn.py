from models.faster_rcnn_model import FasterRCNNModel
from torch.optim import Adam
from dataset import MelanomaDataset, transform
from torch.utils.data import DataLoader

def train_faster_rcnn():
    model = FasterRCNNModel()
    train_dataset = MelanomaDataset("datasets/faster_rcnn_data.json", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    optimizer = Adam(model.model.parameters(), lr=0.001)

    model.train(train_loader, optimizer, epochs=50)

if __name__ == "__main__":
    train_faster_rcnn()
