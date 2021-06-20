import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        num = targets.size(0)
        # 为了防止除0的发生
        smooth = 1

        m1 = inputs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        loss = 1 - score.sum() / num
        return loss
