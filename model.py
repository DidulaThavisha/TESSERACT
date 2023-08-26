import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision

class ResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='vgg19', num_classes=2):
        super(ResNet, self).__init__()
        if name == 'vgg19':
            self.encoder = torchvision.models.vgg19(pretrained=True)
            self.encoder.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.encoder.classifier[6] = nn.Identity()
            self.fc = nn.Linear(4096, num_classes)
        elif name == 'vit':
           
            self.encoder = torchvision.models.vit_b_16(pretrained=True,in_channels=1)
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.encoder.fc = nn.Identity()
            self.fc = nn.Linear(4096, num_classes)
        elif name == 'unet':
            self.encoder = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
            self.encoder.fc =  nn.AdaptiveAvgPool2d(output_size=(1,1))
            #self.encoder.fc = nn.Identity()
            #
            #self.fc =  nn.AdaptiveAvgPool2d(output_size=(1,1))
            self.fc =  nn.Flatten()
            self.fc = nn.Sequential(
                nn.Flatten(),  # Flatten the 2D feature map
                nn.Linear(50176, 128),  # Linear layer with input size 50176 and output size 128
                nn.Sigmoid(),  # Apply ReLU activation
                nn.Linear(128, 6)  # Linear layer with input size 128 and output size 6
)
         
            
            
        else:
            self.encoder = torchvision.models.resnet152(zero_init_residual=True)
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.encoder.fc = nn.Identity()
            self.fc = nn.Linear(2048, num_classes)
    def forward(self, x):

        return self.fc(self.encoder(x))

