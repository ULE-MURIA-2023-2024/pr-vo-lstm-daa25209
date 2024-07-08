import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import VisualOdometryDataset
from model import VisualOdometryModel
from params import *  # Importa todos los parámetros desde params.py

# Crear el modelo de odometría visual
model = VisualOdometryModel(hidden_size, num_layers, bidirectional, lstm_dropout)

transform = T.Compose([
    T.ToTensor(),
    model.resnet_transforms()
])

# Cargar el dataset de entrenamiento
train_dataset = VisualOdometryDataset(
    dataset_path='dataset/train',  # Asegúrate de proporcionar la ruta correcta a tu dataset
    transform=transform,
    sequence_length=sequence_length,
    validation=False
)

# Verificar la longitud del dataset y una muestra de datos
print(f"Longitud del dataset de entrenamiento: {len(train_dataset)}")
if len(train_dataset) > 0:
    sample = train_dataset[0]
    print(f"Muestra de datos: {sample}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Entrenar
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Imprimir información para verificación
print(f"Usando dispositivo: {device}")
print(f"Learning rate: {learning_rate}")
print(f"Batch size: {batch_size}")
print(f"Number of epochs: {epochs}")
print(f"Length of train dataset: {len(train_dataset)}")

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
running_loss = 0.0

for epoch in range(epochs):
    for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        # Asegurarse de que las dimensiones de outputs y labels coinciden
        loss = criterion(outputs, labels[:, -1, :])  # Ajustar las dimensiones de las etiquetas
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader)}")
    running_loss = 0.0

# Guardar los pesos entrenados
torch.save(model.state_dict(), "vo.pt")
