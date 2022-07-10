import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torchvision.models as models
from torch.autograd import Variable

class FedGCNet(nn.Module):
    def __init__(self) -> None:
        pass
    
    def forward(x):
        return x
    

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          s: norm of input feature
          m: additive angular margin
          cos(theta + m)
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
 
        self.in_features = in_features    # 特征输入通道数
        self.out_features = out_features  # 特征输出通道数
        self.s = s                        # 输入特征范数 ||x_i||
        self.m = m                        # 加性角度边距 m (additive angular margin)
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))  # FC 权重
        nn.init.xavier_uniform_(self.weight)  # Xavier 初始化 FC 权重
 
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
 
    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # 分别归一化输入特征 xi 和 FC 权重 W, 二者点乘得到 cosθ, 即预测值 Logit  
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)) # batch_size_features*out_features
        # 由 cosθ 计算相应的 sinθ
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # 展开计算 cos(θ+m) = cosθ*cosm - sinθ*sinm, 其中包含了 Target Logit (cos(θyi+ m)) (由于输入特征 xi 的非真实类也参与了计算, 最后计算新 Logit 时需使用 One-Hot 区别)
        phi = cosine * self.cos_m - sine * self.sin_m  
        # 是否松弛约束??
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # 将 labels 转换为独热编码, 用于区分是否为输入特征 xi 对应的真实类别 yi
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # 计算新 Logit
        #  - 只有输入特征 xi 对应的真实类别 yi (one_hot=1) 采用新 Target Logit cos(θ_yi + m)
        #  - 其余并不对应输入特征 xi 的真实类别的类 (one_hot=0) 则仍保持原 Logit cosθ_j
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # can use torch.where if torch.__version__  > 0.4
        # 使用 s rescale 放缩新 Logit, 以馈入传统 Softmax Loss 计算
        output *= self.s
 
        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResidualBlock,self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
            )
        self.right = shortcut
        
    def forward(self,x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)
        
class ResNet34(nn.Module):  #224x224x3
    def __init__(self, num_classes=10):
        super(ResNet34,self).__init__()
        self.pre = nn.Sequential(
                nn.Conv2d(3,64,7,stride=2,padding=3,bias=False),    # (224+2*padding-kernel)/stride(向下取整)+1，size减半->112
                nn.BatchNorm2d(64),     # 112x112x64
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
                )   #56x56x64
        
        #重复的layer,分别有3,4,6,3个residual block
        self.layer1 = self.make_layer(64,64,3)              #56x56x64,layer1层输入输出一样，make_layer里，应该不用对shortcut进行处理。
        self.layer2 = self.make_layer(64,128,4,stride=2)    #第一个stride=2,剩下3个stride=1;28x28x128
        self.layer3 = self.make_layer(128,256,6,stride=2)   #14x14x256
        self.layer4 = self.make_layer(256,512,3,stride=2)   #7x7x512
        self.fc = nn.Linear(512,num_classes)
        
    def make_layer(self,in_channel,out_channel,block_num,stride=1):
        #当维度增加时，对shortcut进行option B的处理
        shortcut = nn.Sequential(                                   #首个ResidualBlock需要进行option B处理
                nn.Conv2d(in_channel,out_channel,1,stride,bias=False),        #1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
                nn.BatchNorm2d(out_channel)
                )
        layers = []
        layers.append(ResidualBlock(in_channel,out_channel,stride,shortcut))
        
        for i in range(1,block_num):
            layers.append(ResidualBlock(out_channel,out_channel))             #后面的几个ResidualBlock,shortcut直接相加
        return nn.Sequential(*layers)
        
    def forward(self,x):            # 224x224x3
        x = self.pre(x)             # 56x56x64
        x = self.layer1(x)          # 56x56x64
        x = self.layer2(x)          # 28x28x128
        x = self.layer3(x)          # 14x14x256
        x = self.layer4(x)          # 7x7x512
        x = F.avg_pool2d(x,7)       # 1x1x512
        x = x.view(x.size(0),-1)    # flatten: 1x512
        x = self.fc(x)              
        return x

class HCNet(nn.Module):
    def __init__(self, in_num=4, out_num=10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_num, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, out_num)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x

class FocalLoss(nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean', device = 'cpu') -> None:
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                 self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.reduction = reduction
        self.device = device
        
    def forward(self, inputs, target):
        pt = F.softmax(inputs, dim=1)
        class_mask = F.one_hot(target, self.class_num)
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)]
        probs = (pt*class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        alpha = alpha.to(self.device)
        loss = -alpha*torch.pow(1-probs, self.gamma)*log_p
        if self.reduction=='mean':
            loss = loss.mean()
        elif self.reduction=='sum':
            loss = loss.sum()
        return loss

class SpecNet(nn.Module):
    def __init__(self, out_num=10) -> None:
        super().__init__()
        self.res_net = models.resnet18(pretrained=True)
        fc_in_num = self.res_net.fc.in_features
        self.res_net.fc = nn.Linear(fc_in_num, out_num)
        self.dp1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(out_num, out_num)
        self.leaky_relu = nn.LeakyReLU(0.15)
        
    def forward(self, x):
        x = self.res_net(x)
        x = self.dp1(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        return x


class MedTestingNet(nn.Module):
    def __init__(self, base_net = None) -> None:
        super().__init__()
        if base_net is not None:
            assert isinstance(base_net, SpecNet)
        else:
            base_net = SpecNet()
        in_num = base_net.res_net.fc.in_features
        self.backbone = nn.Sequential(*list(base_net.res_net.children())[:-1])
        # frozen pretrained parameters
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        # print(self.backbone)
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_num*2, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.4),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.4),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
        )
        self.fc4 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x1, x2):
        x1 = self.backbone(x1)          # (batch_size, 512, 1, 1)
        x2 = self.backbone(x2)          # (batch_size, 512, 1 ,1)
        x1 = x1.squeeze()               # (batch_size, 512)
        x2 = x2.squeeze()               # (batch_size, 512)
        x = torch.cat((x1, x2), dim=1)  # (batch_size, 1024)
        x = self.fc1(x)                 # (batch_size, 256)
        x = self.fc2(x)                 # (batch_size, 512)
        x = self.fc3(x)                 # (batch_size, 512)
        x = self.fc4(x)                 # (batch_size, 1)
        x = self.sigmoid(x)
        return x.squeeze()


if __name__ =='__main__':
    model = MedTestingNet()
    print(model)
    # model = ArcMarginProduct(10, 10)
    # model = model.to(device='cuda:3')
    