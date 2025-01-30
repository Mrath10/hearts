import torch.nn as nn
from torchvision import models

class FeatureExtraction(nn.Module):
    def __init__(self):
        "For now we try resnet 18"
        
        #Load it
        self.model = models.resnet18(weights=None)

        #Remove fully connected
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        self.embedding = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=0.3),
                nn.Linear(2048,512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 64),
        )

        self.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(64,32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
        )

        # true train all layers
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):

        features = self.model(x)
        embeddings = self.embedding(features)

        logits = self.classifer(embeddings)

        return embeddings, logits


class SiameseNN(nn.Module):
    def __init__(self):
        super(SiameseNN, self).__init__()

        self.feature_extractor = FeatureExtraction()

    def forward(self, x):

            embeddings, logits = self.feature_extractor(x)
            return embeddings, logits
