import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class MelanomaDataset(Dataset):
    def __init__(self, annotation_path, transform=None):
        with open(annotation_path, "r") as f:
            self.annotations = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_path = self.annotations[idx]["image"]
        target = self.annotations[idx]["target"]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target


# Data Augmentation Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
