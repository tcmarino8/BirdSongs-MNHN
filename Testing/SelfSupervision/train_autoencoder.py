
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset import BirdDataset
from models import AutoEncoder
import torch.nn.functional as F



device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = BirdDataset(r"C:\Users\Salle-Cineradio\Documents\MachineLearning\BirdSongs-MNHN\Testing\SelfSupervision\data\train")

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

model = AutoEncoder().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

loss_fn = nn.MSELoss()

for epoch in range(30):

    running = 0

    for images in loader:

        images = images.to(device)

        outputs = model(images)

        loss = loss_fn(outputs, images)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running += loss.item()

    print(epoch, running/len(loader))

torch.save(model.state_dict(),"autoencoder.pt")