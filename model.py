import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
import torchvision.models as models
import torchvision

class ResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='unet', num_classes=2):
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
            self.encoder.fc = nn.Identity()  # Remove the fully connected layer from the encoder
    
            # Create the XGBoost classifier
            self.xgb_classifier = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.001,
                objective='multi:softmax',  # Adjust for your problem type
                num_class=6,  # Number of classes in your problem
                random_state=42
            )
            #self.fc = nn.Linear(224, num_classes)
            #self.encoder = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
            #self.encoder.fc =  nn.AdaptiveAvgPool2d(output_size=(1,1))
            #self.fc =  nn.Flatten()
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(50176, 128),
                nn.Sigmoid(),
                #nn.ReLU(),
                nn.Linear(128, 6)
                #nn.Sigmoid()
            )
                #nn.Flatten(),  # Flatten the 2D feature map
                #nn.Linear(50176, 224), 
                #nn.BatchNorm1d(224),# Linear layer with input size 50176 and output size 128
                #nn.ReLU(),
                ##nn.Dropout(0.5),# Apply ReLU activation
                #nn.Linear(224, 224),
                #nn.BatchNorm1d(224),
                #nn.Linear(224, 6),
                #nn.Sigmoid()# Linear layer with input size 128 and output size 6
                

         
            
            
        else:
            self.encoder = torchvision.models.resnet152(zero_init_residual=True)
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.encoder.fc = nn.Identity()
            self.fc = nn.Linear(2048, num_classes)
    #def forward(self, x):
    def forward(self, x):
        # Forward pass through the UNet encoder
        features = self.encoder(x)
        features_np = features# Convert to numpy array

        return self.fc(features_np) # Return extracted features for XGBoost
        #return self.fc(self.encoder(x))

