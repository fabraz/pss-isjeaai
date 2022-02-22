import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import re


class EffModule(nn.Module):
    '''
    Neural net module that receives image(s) as input and
    returns a prob tensor with the number of classes.
    '''

    def __init__(self,
                 n_classes=2,
                 n_inputs=1,
                 skip_freezing_batch_norm=True,
                 stop_freezing_at=14,
                 pretrained=True,
                 efficientnet_name='efficientnet-b0'):
        super(EffModule, self).__init__()
        self.base_model = EfficientNet.from_pretrained(
            efficientnet_name) if pretrained else EfficientNet.from_name(
                efficientnet_name)
        self.stop_freezing_at = stop_freezing_at
        self.skip_freezing_batch_norm = skip_freezing_batch_norm
        self.freeze()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
        )

        self.final = nn.Sequential(
            nn.Linear(1280 * n_inputs, n_classes),
            nn.Softmax(dim=1)
        )

    def freeze(self):
        off_training = ['_conv_stem', '_blocks']
        if not self.skip_freezing_batch_norm:
            off_training.append('_bn0')
        for name, child in self.base_model.named_children():
            if name not in off_training:
                trainable = True
            else:
                trainable = False
            for child_name, child_param in child.named_parameters():
                child_param.requires_grad = trainable
                match = re.match(r"^(\d+)\._(\w{2}).*$", child_name)
                if match:
                    number = int(match[1])
                    start = match[2]
                    if number >= self.stop_freezing_at:
                        child_param.requires_grad = True
                    else:
                        child_param.requires_grad = False
                        if start == 'bn' and self.skip_freezing_batch_norm:
                            child_param.requires_grad = True

    def forward(self):
        pass


class Eff_3(EffModule):
    def __init__(self,
                 n_classes,
                 skip_freezing_batch_norm=True,
                 stop_freezing_at=11,
                 pretrained=True):
        super(Eff_3, self).__init__(n_classes,
                                                  3,
                                                  skip_freezing_batch_norm,
                                                  stop_freezing_at,
                                                  pretrained)

    def forward(self, prev_page, targ_page, next_page):
        x1 = self.base_model.extract_features(prev_page)
        pooled_features_1 = torch.nn.functional.adaptive_avg_pool2d(x1, 1)
        x1 = self.classifier(pooled_features_1)

        x2 = self.base_model.extract_features(targ_page)
        pooled_features_2 = torch.nn.functional.adaptive_avg_pool2d(x2, 1)
        x2 = self.classifier(pooled_features_2)

        x3 = self.base_model.extract_features(next_page)
        pooled_features_3 = torch.nn.functional.adaptive_avg_pool2d(x3, 1)
        x3 = self.classifier(pooled_features_3)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.final(x)
        return x

    
class Eff_2(EffModule):
    def __init__(self, 
                 n_classes,
                 skip_freezing_batch_norm=True, 
                 stop_freezing_at=11,
                 pretrained=True):
        super(Eff_2, self).__init__(n_classes, 
                                                2, 
                                                skip_freezing_batch_norm, 
                                                stop_freezing_at,
                                                pretrained)
        
    def forward(self, prev_page, targ_page):
        x1 = self.base_model.extract_features(prev_page)
        pooled_features_1 = torch.nn.functional.adaptive_avg_pool2d(x1, 1)                
        x1 = self.classifier(pooled_features_1)

        x2 = self.base_model.extract_features(targ_page)
        pooled_features_2 = torch.nn.functional.adaptive_avg_pool2d(x2, 1)                        
        x2 = self.classifier(pooled_features_2)

        x = torch.cat((x1, x2), dim=1)
        x = self.final(x)
        return x
    
class Eff_1(EffModule):
    def __init__(self, 
                 n_classes,
                 skip_freezing_batch_norm=True, 
                 stop_freezing_at=11,
                 pretrained=True):
        super(Eff_1, self).__init__(n_classes, 
                                               1, 
                                               skip_freezing_batch_norm, 
                                               stop_freezing_at,
                                               pretrained)
        
    def forward(self, x):
        x1 = self.base_model.extract_features(x)
        pooled_features = torch.nn.functional.adaptive_avg_pool2d(x1, 1)                
        x = self.classifier(pooled_features)
        x = self.final(x)
        return x
    
    
class VGG16Module(nn.Module):
    '''
    VGG16 Neural net module that receives image(s) as input and returns a prob tensor with the number of classes. 
    '''
    def __init__(self, 
                 n_classes, 
                 n_pages, 
                 stop_freezing_at=28,
                 pretrained=True):
        super().__init__()
        
        self.base_model = models.vgg16(pretrained=pretrained)
        
        self.features = self.base_model.features
        
        for name, parameter in self.features.named_parameters():
            parameter.requires_grad = False
            match = re.match(r"^(\d+).*$", name)
            if match:
                name_start = int(match[1])
                if name_start >= stop_freezing_at:
                    parameter.requires_grad = True        
        
        self.avgpool = nn.AdaptiveAvgPool2d(7)   

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
                
        self.final = nn.Sequential(
            nn.Linear(512 * n_pages , 256),
            nn.LeakyReLU(), 
            nn.Linear(256, n_classes),
            nn.LeakyReLU(),
            nn.Softmax(dim=1)
        )
    
class VGG16_1(VGG16Module):
    def __init__(self, 
                 n_classes, 
                 stop_freezing_at=28,
                 pretrained=True):
        super(VGG16_1, self).__init__(n_classes, 
                                                 1, 
                                                 stop_freezing_at,
                                                 pretrained)
        
    def forward(self, x):
        x1 = self.features(x)
        x = self.classifier(x1)
        x = self.final(x)
        return x
    
class VGG16_2(VGG16Module):
    def __init__(self, 
                 n_classes, 
                 stop_freezing_at=28,
                 pretrained=True):
        super(VGG16_2, self).__init__(n_classes, 
                                                  2, 
                                                  stop_freezing_at,
                                                  pretrained)
        
    def forward(self, prevPage, targPage):
        x1 = self.features(prevPage)
        x1 = self.classifier(x1)
        
        x2 = self.features(targPage)
        x2 = self.classifier(x2) 
        
        x = torch.cat((x1, x2), dim=1)        
        x = self.final(x)
        
        return x
    
class VGG16_3(VGG16Module):
    def __init__(self, 
                 n_classes, 
                 stop_freezing_at=28,
                 pretrained=True):
        super(VGG16_3, self).__init__(n_classes, 
                                                    3, 
                                                    stop_freezing_at,
                                                   pretrained)

    def forward(self, prev_page, targ_page, next_page):
        x1 = self.features(prev_page)
        x1 = self.classifier(x1)
        
        x2 = self.features(targ_page)
        x2 = self.classifier(x2)        

        x3 = self.features(next_page)
        x3 = self.classifier(x3)                
        
        x = torch.cat((x1, x2, x3), dim=1)        
        x = self.final(x)
        
        return x    