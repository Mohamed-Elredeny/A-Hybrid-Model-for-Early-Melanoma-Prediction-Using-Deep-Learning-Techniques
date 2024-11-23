import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T

class FasterRCNNModel:
    def __init__(self):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        num_classes = 2  # For melanoma detection
        self.model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)

    def train(self, train_loader, optimizer, epochs):
        self.model.train()
        for epoch in range(epochs):
            for images, targets in train_loader:
                optimizer.zero_grad()
                loss_dict = self.model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
                loss.backward()
                optimizer.step()
        torch.save(self.model.state_dict(), "trained_faster_rcnn.pth")
        print("Faster R-CNN training complete.")

    def predict(self, images):
        self.model.eval()
        return self.model(images)

    def evaluate(self, val_loader):
        self.model.eval()
        results = []
        with torch.no_grad():
            for images, targets in val_loader:
                predictions = self.model(images)
                results.append((predictions, targets))
        return results
