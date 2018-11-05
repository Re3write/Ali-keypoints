from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.size())
        b, c, _, _ = x.size()
        # print('参数'+str(b),str(c))
        y = self.avg_pool(x).view(b, c)
        # print(y.size())
        y = self.fc(y).view(b, c, 1, 1)

        return x * y