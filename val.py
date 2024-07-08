
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import VisualOdometryDataset
from model import VisualOdometryModel
from params import *


# Create the visual odometry model
model = VisualOdometryModel(hidden_size, num_layers, bidirectional, lstm_dropout)

transform = T.Compose([
    T.ToTensor(),
    model.resnet_transforms()
])

# Load dataset
val_dataset = VisualOdometryDataset(
    dataset_path='ruta/a/tu/dataset',  # Asegúrate de proporcionar la ruta correcta a tu dataset
    transform=transform,
    sequence_length=sequence_length,
    validation=True
)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Validate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)
model.load_state_dict(torch.load("vo.pt"))
model.eval()

validation_string = ""

with torch.no_grad():
    for images, labels, timestamp in tqdm(val_loader, desc="Validating:"):
        images = images.to(device)
        labels = labels.to(device)

        target = model(images).cpu().numpy().tolist()[0]

        # Concatenar los resultados en validation_string
        validation_string += f"{timestamp.item()} {' '.join(map(str, target))}\n"

# Guardar los resultados en un archivo
with open("validation.txt", "a") as f:
    f.write(validation_string)
