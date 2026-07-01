import torch
import matplotlib.pyplot as plt
from dataset import BirdDataset
from models import AutoEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoEncoder().to(device)
model.load_state_dict(torch.load("autoencoder.pt"))
model.eval()

dataset = BirdDataset("data/train")
img = dataset[0].unsqueeze(0).to(device)

with torch.no_grad():
    features = model.encoder(img)

# take first channel of feature map
feat = features[0,0].cpu()

plt.imshow(feat, cmap="magma")  
plt.title("Encoder feature activation")
plt.show()