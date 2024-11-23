import torch

class MelanoHybridModel(torch.nn.Module):
    def __init__(self, yolo_model, faster_rcnn_model):
        super(MelanoHybridModel, self).__init__()
        self.yolo = yolo_model
        self.faster_rcnn = faster_rcnn_model

    def forward(self, images):
        yolo_preds = self.yolo(images)
        rcnn_preds = self.faster_rcnn(images)
        return self.fuse_predictions(yolo_preds, rcnn_preds)

    def fuse_predictions(self, yolo_preds, rcnn_preds):
        combined = {
            "boxes": torch.cat((yolo_preds["boxes"], rcnn_preds["boxes"]), dim=0),
            "scores": torch.cat((yolo_preds["scores"], rcnn_preds["scores"]), dim=0),
            "labels": torch.cat((yolo_preds["labels"], rcnn_preds["labels"]), dim=0)
        }
        return combined
