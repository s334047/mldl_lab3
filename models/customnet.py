import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1) #layer in più per arrivare a 256 canali
        
        self.pool = nn.MaxPool2d(2) #pooling per ridurre la dimensione spaziale
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #pooling globale per ridurre la dimensione spaziale
        self.fc1 = nn.Linear(256, 200) #fully connected layer (200 = n di classi in TinyImageNet)

    def forward(self, x):
        # Define forward pass

        # B x 3 x 224 x 224
        x = self.conv1(x).relu() # B x 64 x 224 x 224
        x = self.pool(x)         # B x 64 x 112 x 112

        x = self.conv2(x).relu() # B x
        x = self.pool(x)         # B x

        x = self.conv3(x).relu() # B x
        x = self.pool(x)         # B x

        x = self.avgpool(x)      # B x

        x = torch.flatten(x, 1)  # B x
        x = self.fc1(x)

        return x
