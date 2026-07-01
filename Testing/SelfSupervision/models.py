# models.py

import torch
import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(128,64,2,stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(64,32,2,stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(32,1,2,stride=2),
            nn.Sigmoid()
        )

    def forward(self,x):

        z = self.encoder(x)

        out = self.decoder(z)

        return out