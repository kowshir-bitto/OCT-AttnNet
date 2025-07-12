import torch.nn as nn
from torchvision import models
from .attention_modules import BAM, ECA

class CustomImageClassifier(nn.Module):
    def __init__(self, model_architecture, num_classes, pretrained=True, freeze_features=True):
        super(CustomImageClassifier, self).__init__()

        if model_architecture != models.inception_v3:
            raise ValueError("This model only supports InceptionV3 with attention modules.")

        weights_enum = models.Inception_V3_Weights.DEFAULT if pretrained else None
        self.model = model_architecture(weights=weights_enum)

        if freeze_features:
            for param in self.model.parameters():
                param.requires_grad = False

        self._add_attention_modules_inception(self.model)

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        if hasattr(self.model, 'fc'):
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)

    def _add_attention_modules_inception(self, model):
        in_channels = model.Mixed_7c.branch_pool.conv.out_channels
        model.Mixed_7c.add_module('bam', BAM(in_channels))
        model.Mixed_7c.add_module('eca', ECA(in_channels))

    def forward(self, x):
        return self.model(x)

def build_model(model_config, num_classes):
    model = CustomImageClassifier(
        model_architecture=model_config["model_architecture"],
        num_classes=num_classes,
        pretrained=model_config["pretrained"],
        freeze_features=model_config["freeze_features"]
    )
    return model
