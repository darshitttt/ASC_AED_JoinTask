import torch
import torch.nn as nn

torch.manual_seed(13)
torch.cuda.manual_seed(13)

class logmelAE_30secs(nn.Module):

    def __init__(self):
        super(logmelAE_30secs, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(25, 50, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(50, 5, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(5, 50, kernel_size=(3,2), stride=2, padding=0),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(50, 25, kernel_size=(2,2), stride=2, padding=0),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(25, 1, kernel_size=(2,3), stride=2, padding=2),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x