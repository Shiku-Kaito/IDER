import torch
import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ST_net(nn.Module):
    def __init__(self, args, num_outputs):
        super().__init__()
        self.feature_extractor = Mlp(in_features=512, hidden_features=512, out_features=512)
        self.regressor = nn.Linear(512, num_outputs)

    def forward(self, img):
        img_feat = self.feature_extractor(img) # B x 512 x 7 x 7
        y = self.regressor(img_feat)
        if y.shape[1]!=1:
            y = y.squeeze()
        return {"y": y}


