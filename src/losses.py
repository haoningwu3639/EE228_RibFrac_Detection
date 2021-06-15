import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, weight = None, size_average = True, alpha = 1, gamma = 0.3):
        super(SoftDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1
        
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2
        
        score = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        
        ce = F.binary_cross_entropy_with_logits(logits, targets)
        fc = self.alpha * (1 - torch.exp(-ce)) ** self.gamma * ce
        
        loss = (score + fc) / 2
        
        return loss
        

class MixLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x, y):
        lf, lfw = [], []
        for i, v in enumerate(self.args):
            if i % 2 == 0:
                lf.append(v)
            else:
                lfw.append(v)
        mix = sum([w * l(x, y) for l, w in zip(lf, lfw)])
        return mix


class DiceLoss(nn.Module):
    def __init__(self, image=False):
        super().__init__()
        self.image = image

    def forward(self, x, y):
        x = x.sigmoid()
        i, u = [t.flatten(1).sum(1) if self.image else t.sum() for t in [x * y, x + y]]

        dice = (2 * i + 1) / (u + 1)
        dice = 1 - dice.mean()
        return dice
