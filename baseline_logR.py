"""
baseline_logR.py
-------------------
This is a script to train/test a logistic regression model
on our dataset. For this baseline, our preprocessing consisted
of simply flattening the image into a (3 * 4032 * 3024) length
vector before passing it into the model.
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import classification_report
from utils.preprocessing import flatten

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = Compose([
    Resize((3024, 4032)),
    ToTensor(),
])

data_dir = './img'
dataset = ImageFolder(data_dir, transform=transform)

# split into train and eval
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.linear(x)

num_classes = len(dataset.classes)
input_size = 3 * 3024 * 4032  # flattened image size

model = LogisticRegressionModel(input_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
tolerance = 1e-4
previous_loss = float('inf')
max_iterations = 100
iteration = 0

print("training with mini batch GD...")
while iteration < max_iterations:
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader, leave=False):
        images = flatten(images).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        print(f'Iteration {iteration + 1}, Train Loss: {train_loss:.4f}')
        
        # convergence check
        if abs(previous_loss - train_loss) < tolerance:
            print("Convergence achieved!")
            break
        previous_loss = train_loss
        iteration += 1

# validation and reporting
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in val_loader:
        images = flatten(images).to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=dataset.classes))
