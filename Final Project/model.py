import torch.nn as nn
import torchvision.models as models

class VGG16(nn.Module):
    def __init__(self,num_class):
        super(VGG16,self).__init__()
        self.model = models.vgg16(pretrained=True)

        for param in self.model.features.parameters():
            param.require_grad = False

        num_features = self.model.classifier[6].in_features
        features = list(self.model.classifier.children())[:-1]
        features.extend([nn.Linear(num_features, 15)])
        self.model.classifier = nn.Sequential(*features)

    def forward(self,X):
        out=self.model(X)
        return out


class ResNet34(nn.Module):
    def __init__(self,num_class):
        super(ResNet34,self).__init__()
        self.model = models.resnet34(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad=False
            
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 15)

    def forward(self,X):
        out=self.model(X)
        return out