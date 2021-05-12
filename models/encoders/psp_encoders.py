import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module

from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE
from models.stylegan2.model import EqualLinear

from vit_pytorch import ViTV2

class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)

        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = opts.style_num
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        
        return out

class GradualStyleEncoderV2(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleEncoderV2, self).__init__()
        
        self.layer1 =  EqualLinear(38, 512, lr_mul=1)
        self.layer2 =  EqualLinear(512, 512, lr_mul=1)
        self.layer3 =  EqualLinear(512, 512, lr_mul=1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x*0.01
        #x = torch.tanh(x)
        out = x.unsqueeze(1).repeat(1,18,1)
        
        return out
    
class PointNetEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(PointNetEncoder, self).__init__()
        
        self.grid = torch.from_numpy(np.array(np.meshgrid(np.linspace(-1,1,256), np.linspace(-1,1,256), sparse=False)).astype(np.float32)).cuda()
        self.convs = [torch.nn.Conv1d(2, 64, 1),PReLU(64)]
        self.style_fcs = [EqualLinear(64*opts.label_nc, 512, lr_mul=1)]
        for i in range(17):
            self.convs.extend([torch.nn.Conv1d(64, 64, 1),PReLU(64)])
            self.style_fcs.extend([EqualLinear(64*opts.label_nc, 512, lr_mul=1)])
    
        self.convs = Sequential(*self.convs)
        self.style_fcs = Sequential(*self.style_fcs)
        
    def forward(self, x):
        
        z_list_batch = []
        for b in range(x.shape[0]):
            c_poolz_list = []
            for c in range(x.shape[1]):
                points = self.grid[:,x[b,c]==1]
                if points.shape[1] == 0:
                    points = torch.zeros(2,1).cuda()

                points = points.unsqueeze(0)
                z = points
                poolz_list = []
                for i in range(18):
                    z = self.convs[i](z)
                    poolz = torch.max(z, 2, keepdim=True)[0]
                    poolz = poolz.view(1,1,64)
                    poolz_list.append(poolz)
                poolz_list = torch.cat(poolz_list,dim=0)
                c_poolz_list.append(poolz_list)

            c_poolz_list = torch.cat(c_poolz_list,dim=2)
            stylez_list = []
            for i in reversed(range(18)):
                style_z = self.style_fcs[i](c_poolz_list[i])
                stylez_list.append(style_z.unsqueeze(0))
            stylez_list = torch.cat(stylez_list,dim=1)
            z_list_batch.append(stylez_list)
        out = torch.cat(z_list_batch,dim=0)
        
        return out

class PointNetEncoderV2(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(PointNetEncoderV2, self).__init__()
        
        self.grid = torch.from_numpy(np.array(np.meshgrid(np.linspace(-1,1,256), np.linspace(-1,1,256), sparse=False)).astype(np.float32)).cuda()
        self.conv1 = torch.nn.Conv1d(2, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc1= EqualLinear(64*opts.label_nc, 512, lr_mul=1)

    def forward(self, x):
        z_list_batch = []
        for b in range(x.shape[0]):
            z_list = []
            for c in range(x.shape[1]):
                points = self.grid[:,x[b,c]==1]
                if points.shape[1] == 0:
                    points = torch.zeros(2,1).cuda()

                points = points.unsqueeze(0)
                if points.shape[2] != 1:
                    z = F.leaky_relu(self.conv1(points))
                    z = F.leaky_relu(self.conv2(z))
                    z = F.leaky_relu(self.conv3(z))
                else:
                    z = F.leaky_relu(self.conv1(points))
                    z = F.leaky_relu(self.conv2(z))
                    z = F.leaky_relu(self.conv3(z))
                    
                z = torch.max(z, 2, keepdim=True)[0]
                z = z.view(64)
                z_list.append(z)
            z_list = torch.cat(z_list).unsqueeze(0)
            z_list = self.fc1(z_list)
            z_list_batch.append(z_list)
        z_list_batch = torch.cat(z_list_batch)
        out = z_list_batch.unsqueeze(1).repeat(1,18,1)
        return out
    
class TransformerEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(TransformerEncoder, self).__init__()
        
        self.v = ViTV2(
            image_size = 256,
            patch_size = 32,
            num_classes = 512,
            channels = opts.label_nc,
            dim = 1024,
            depth = 6,
            heads = 8,
            mlp_dim = 2048,
            attn_dropout = 0.1,
            ff_dropout = 0.1
        )

    def forward(self, x):
        x = self.v(x)
        x = x.unsqueeze(1).repeat(1,18,1)
        return x

class BackboneEncoderUsingLastLayerIntoW(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoW')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x


class BackboneEncoderUsingLastLayerIntoWPlus(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoWPlus, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoWPlus')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer_2 = Sequential(BatchNorm2d(512),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(512 * 7 * 7, 512))
        self.linear = EqualLinear(512, 512 * 18, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer_2(x)
        x = self.linear(x)
        x = x.view(-1, 18, 512)
        return x
