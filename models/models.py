import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(13)
torch.cuda.manual_seed(13)

class ASC_Model00(nn.Module):

    """
    Constructor method for ASC_Model00 class initializing convolutional, pooling, fully connected, and output layers."""
    def __init__(self, return_second_last=False):
        super(ASC_Model00, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        
        # Pooling layer
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.relu = nn.ReLU()
        # Output layers
        self.output_layer = nn.Linear(2048, 10)
        self.softmax = nn.Softmax()
        self.return_second_last = return_second_last
        #self.second_last_layer = nn.Linear(2048, 2048)
        
    def forward(self, x):
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Pooling layer
        x = self.pool(x)
        
        # Flatten tensor
        x = x.flatten(start_dim=1)
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        second_last = self.relu(self.fc3(x))
        
        # Output layers
        output = self.output_layer(second_last)
        #second_last = self.second_last_layer(x)
        if self.return_second_last:
            return (output), second_last
        else:
            return output

class logmelAE_30secs(nn.Module):

    def __init__(self):
        super(logmelAE_30secs, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(25, 50, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(50, 100, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(100, 5, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(5, 100, kernel_size=(2,2), stride=2, padding=0),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(100, 50, kernel_size=(2,2), stride=2, padding=0),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(50, 25, kernel_size=(2,2), stride=2, padding=(1,0)),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(25, 1, kernel_size=(2,3), stride=2, padding=2),
            nn.LeakyReLU()
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
    

def train(model, device, train_loader, loss_fn, optimizer, epoch):
    model.train()
    model = model.to(device)
    
    for batch_idx, batch in enumerate(train_loader):
        data, target = batch['data'].to(device), batch['scene_label'].to(device, dtype=torch.float)
        optimizer.zero_grad()
        output = (model(data))
        
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    return model

            