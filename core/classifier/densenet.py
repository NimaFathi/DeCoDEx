import torch
import torchvision
import torch.nn as nn
from torchvision.models import DenseNet121_Weights
from torch.nn import functional as F

class DenseNet121(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torchvision.models.densenet121()
        kernel_count = self.net.classifier.in_features
        self.net.classifier = nn.Linear(kernel_count, 1)
        
    
    def forward(self, x):
        return self.net(x)


class ClassificationModel(torch.nn.Module):
    def __init__(self, path_to_weights):

        super().__init__()
        self.net = DenseNet121()

        # load the model from the checkpoint
        state_dict = torch.load(path_to_weights, map_location='cpu')
        
        self.net.load_state_dict(state_dict['model'])

    def forward(self, x):
        x = self.net(x)
        return x


class TwoHeadedDenseNet(nn.Module):
    def __init__(self, num_classes=1, selected_head=0):
        super(TwoHeadedDenseNet, self).__init__()
        self.backbone = torchvision.models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        num_features = self.backbone.classifier.in_features
        self.head1 = nn.Linear(num_features, num_classes)  # Classifier head
        self.head2 = nn.Linear(num_features, num_classes)  # Detector head
        self.selected_head = selected_head  # Determine which head to use  
        
    def forward(self, x):
        features = self.backbone.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        if self.selected_head == 0:
            out = self.head1(out)
        else:
            out = self.head2(out)
        return out
    
class SharedClassifier(torch.nn.Module):
    def __init__(self, path_to_weights):

        super().__init__()
        self.net = TwoHeadedDenseNet(selected_head=0)

        # load the model from the checkpoint
        state_dict = torch.load(path_to_weights, map_location='cpu')
        
        self.net.load_state_dict(state_dict['model'])

    def forward(self, x):
        x = self.net(x)
        return x
    
class SharedDetector(torch.nn.Module):
    def __init__(self, path_to_weights):

        super().__init__()
        self.net = TwoHeadedDenseNet(selected_head=1)

        # load the model from the checkpoint
        state_dict = torch.load(path_to_weights, map_location='cpu')
        
        self.net.load_state_dict(state_dict['model'])

    def forward(self, x):
        x = self.net(x)
        return x