"""
vit-test.py
--------------------------
This script runs testing and prints results (including f1 score, precision, 
and recall) on the model that is saved after running vit.py. 
"""
# Change this filename to the model you get out of vit.py.
MODEL_FILENAME = 'models/model-no-red.pth'

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report
import timm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = './img'
dataset = ImageFolder(data_dir, transform=transform)

# extract 20% of images for validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# load pre-trained model
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=len(dataset.classes))
model.load_state_dict(torch.load(MODEL_FILENAME))
model = model.to(device)
model.eval()

all_labels = []
all_predictions = []

# run testing
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

print("Classification Report:")
print(classification_report(all_labels, all_predictions, target_names=dataset.classes, digits=4))
