import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter as P

from resnet import resnet34, resnet50
# from models.densenet import DenseNet121, DenseNet201
import torchvision.models as models


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B x (N) x C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B x C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B x (N) x (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B x C x N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
class Attention(nn.Module):
    def __init__(self, ch, which_conv=nn.Conv2d, name='attention'):
        super(Attention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


class ResNet50_Attn(nn.Module):
    def __init__(self, pretrained=False, out_features=125):
        super(ResNet50_Attn, self).__init__()
        self.model = resnet50(pretrained=pretrained)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(in_features=2048, out_features=out_features)
        self.attn = Attention(2048)

    def forward(self, x):
        # shape [N, C, H, W]
        x = self.model(x)

        attn_feature = self.attn(x)

        attn_feature = self.attn_avgpool(attn_feature)
        attn_feature = attn_feature.view(attn_feature.size(0), -1)

        x = self.fc(attn_feature)
        return x


class ResNet50_Self_Attn(nn.Module):
    def __init__(self, pretrained=False, out_features=125):
        super(ResNet50_Self_Attn, self).__init__()
        self.model = resnet50(pretrained=pretrained)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(in_features=2048, out_features=out_features)
        self.attn = Self_Attn(2048, 'relu')

    def forward(self, x):
        # shape [N, C, H, W]
        x = self.model(x)

        attn_feature, p = self.attn(x)
        attn_feature = self.avgpool(attn_feature)
        attn_feature = attn_feature.view(attn_feature.size(0), -1)

        x = self.fc(attn_feature)
        return x


from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=28, patch_size=7, in_chans=512, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class ResNet50_AgentAttn(nn.Module):
    def __init__(self, pretrained=False, out_features=7):
        super(ResNet50_AgentAttn, self).__init__()
        self.model = resnet50(pretrained=pretrained)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(in_features=87806, out_features=out_features)  # 2048
        # Replace Self_Attn with AgentAttention
        self.attn = AgentAttention(2048, num_patches=49, num_heads=8, agent_num=49)

    def forward(self, x):
        # shape [N, C, H, W]
        x = self.model(x)

        # Apply AgentAttention instead of Self_Attn
        attn_feature = self.attn(x, H=x.size(2), W=x.size(3))

        attn_feature = self.avgpool(attn_feature)
        attn_feature = attn_feature.view(attn_feature.size(0), -1)

        x = self.fc(attn_feature)
        return x


class ECA_Layer(nn.Module):
    """Constructs a ECA module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=3):
        super(ECA_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ResNet50_ECA(nn.Module):
    def __init__(self, pretrained=False, out_features=7):
        super(ResNet50_ECA, self).__init__()
        self.model = resnet50(pretrained=pretrained)
        self.attn = ECA_Layer(2048)
        # 修改模型的残差块，添加ECA模块
        for layer in [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]:
            for block in layer:
                # 获取Bottleneck模块的最后一个卷积层的输出通道数
                out_channels = block.conv3.out_channels
                # 添加ECA模块
                block.add_module('eca', ECA_Layer(out_channels))

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(in_features=2048, out_features=out_features)

    def forward(self, x):
        # shape [N, C, H, W]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        # ECA模块会在每个残差块的前向传播过程中自动被调用

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class DenseNet121_Attn(nn.Module):
    def __init__(self, pretrained=False, out_features=125):
        super(DenseNet121_Attn, self).__init__()
        self.model = DenseNet121(pretrained=pretrained)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(in_features=2048, out_features=out_features)
        self.attn = Attention(2048)

    def forward(self, x):
        # shape [N, C, H, W]
        x = self.model(x)

        attn_feature = self.attn(x)

        attn_feature = self.attn_avgpool(attn_feature)
        attn_feature = attn_feature.view(attn_feature.size(0), -1)

        x = self.fc(attn_feature)
        return x


class DenseNet121_Self_Attn(nn.Module):
    def __init__(self, pretrained=False, out_features=125):
        super(DenseNet121_Self_Attn, self).__init__()
        # 加载预训练的DenseNet121模型
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT).features

        # 注意：DenseNet的输出特征图大小与ResNet不同，可能需要调整avgpool的kernel_size
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 使用AdaptiveAvgPool2d以适应不同的输入尺寸
        self.fc = nn.Linear(in_features=1024, out_features=out_features)  # DenseNet121的输出通道数为1024

        # 假设Self_Attn模块接受1024通道的输入
        self.attn = Self_Attn(1024, 'relu')  # 修改输入通道数以匹配DenseNet的输出

    def forward(self, x):
        # shape [N, C, H, W]
        x = self.model(x)  # 输出为特征图

        attn_feature, p = self.attn(x)

        attn_feature = self.avgpool(attn_feature)
        attn_feature = attn_feature.view(attn_feature.size(0), -1)

        x = self.fc(attn_feature)
        return x


class DenseNet121_CA_tail(nn.Module):
    def __init__(self, pretrained=False, out_features=125):
        super(DenseNet121_CA_tail, self).__init__()
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT).features
        self.attn = CoordinateAttention(1024, 1024)
        self.maxpool = nn.AdaptiveMaxPool2d(1)  # 添加最大池化层
        self.dropout = nn.Dropout(p=0.5)  # 添加Dropout层
        self.fc = nn.Linear(in_features=1024, out_features=out_features)

    def forward(self, x):
        x = self.model(x)
        attn_feature = self.attn(x)
        attn_feature = self.maxpool(attn_feature)  # 使用最大池化
        attn_feature = attn_feature.view(attn_feature.size(0), -1)
        x = self.fc(attn_feature)
        return x


import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def remove_fc(state_dict):
    """Remove the fc layer parameters from state_dict, as well as layer3 and layer4 parameters."""
    keys_to_remove = [key for key in state_dict.keys() if key.startswith(('fc.'))]  # , 'layer3.', 'layer4.'
    for key in keys_to_remove:
        del state_dict[key]
    return state_dict


class AgentAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, agent_num=49, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_patches = num_patches
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0] // sr_ratio, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1] // sr_ratio))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W):
        # 确保x的形状是(batch size, number of patches, channels)
        # x = x.flatten(2).transpose(1, 2)
        b, n, c = x.shape
        # print(x.shape)
        num_heads = self.num_heads
        head_dim = c // num_heads
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        agent_tokens = self.pool(q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        kv_size = (self.window_size[0] // self.sr_ratio, self.window_size[1] // self.sr_ratio)
        position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, H, W, c).permute(0, 3, 1, 2)
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ResNet50_Self_Attn3(nn.Module):
    def __init__(self, pretrained=False, out_features=125):
        super(ResNet50_Self_Attn3, self).__init__()

        # 加载预训练的 ResNet50，但不包括最后一层的全连接层
        # resnet = resnet50(weights=None)  # 初始化时不加载任何预训练权重
        resnet = models.resnet50(weights=None)
        '''
        if pretrained:
            # 获取预训练权重，然后从中移除不需要的层的权重
            pretrained_weights = ResNet50_Weights.DEFAULT.get_state_dict(progress=True)
            pretrained_weights = remove_fc(pretrained_weights)
            resnet.load_state_dict(pretrained_weights, strict=False)
        resnet = resnet50(pretrained=False)  
        '''
        if pretrained:
            # 加载预训练权重，然后从中移除不需要的层的权重
            pretrained_weights = torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/resnet50-19c8e357.pth',
                progress=True)
            pretrained_weights = remove_fc(pretrained_weights)
            resnet.load_state_dict(pretrained_weights, strict=False)

        # 定义 ResNet50 的前几层
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2

        # self.attn = AgentAttention(2048, num_patches=49, num_heads=8, agent_num=49)
        # self.PatchEmbed = PatchEmbed(img_size=14,patch_size=2,in_chans=1024,embed_dim=2048)#改
        self.attn = AgentAttention(512, num_patches=196, num_heads=8, agent_num=49)
        self.PatchEmbed = PatchEmbed(img_size=28, patch_size=2, in_chans=512, embed_dim=512)
        # 添加注意力机制
        # self.attn = Self_Attn(512, 'relu')  # 注意力机制的输入通道数为 512

        # self.attn=ECA_Layer(512)
        # 定义 ResNet50 的剩余层
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, out_features)

    def forward(self, x):
        # 通过 ResNet50 的前几层
        B = x.shape[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # 应用注意力机制
        # attn_feature, _ = self.attn(x)

        patch_embed, (H, W) = self.PatchEmbed(x)
        attn_feature = self.attn(patch_embed, H=14, W=14)  # 原来是7，7
        # x = attn_feature.reshape(-1, 7, 7, 2048).permute(0, 3, 1, 2)#原来有
        # 通过 ResNet50 的剩余层
        attn_feature = attn_feature.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.layer3(attn_feature)
        # import pdb
        # pdb.set_trace()
        # x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

