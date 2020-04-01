from core import *
from torch_backend import *
from td import Conv2d_TD, Linear_TD, Conv2d_col_TD

# Network definition
def conv_bn_TD(c_in, c_out, gamma=0.0, alpha=0.0, block_size=16):
    return {
        # 'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False), 
        'conv': Conv2d_TD(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False, gamma=gamma, alpha=alpha, block_size=block_size), 
        'bn': BatchNorm(c_out), 
        'relu': nn.ReLU(True)
    }

def conv_bn(c_in, c_out):
    return {
        'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False), 
        'bn': BatchNorm(c_out), 
        'relu': nn.ReLU(True)
    }

def residual(c):
    return {
        'in': Identity(),
        'res1': conv_bn(c, c),
        'res2': conv_bn(c, c),
        'add': (Add(), ['in', 'res2/relu']),
    }

def net_low(channels=None, weight=0.125, pool=nn.MaxPool2d(2), extra_layers=(), res_layers=('layer1', 'layer3'), gamma=0.0, alpha=0.0, block_size=16):
    channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
    n = {
        'input': (None, []),
        'prep': conv_bn(3, channels['prep']),
        'layer1': dict(conv_bn_TD(channels['prep'], channels['layer1'], gamma=gamma, alpha=alpha, block_size=block_size), pool=pool),
        'layer2': dict(conv_bn_TD(channels['layer1'], channels['layer2'], gamma=gamma, alpha=alpha, block_size=block_size), pool=pool),
        'layer3': dict(conv_bn_TD(channels['layer2'], channels['layer3'], gamma=gamma, alpha=alpha, block_size=block_size), pool=pool),
        'pool': nn.MaxPool2d(4),
        'flatten': Flatten(),
        'linear': nn.Linear(channels['layer3'], 10, bias=False),
        'logits': Mul(weight),
    }
    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer])
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer])       
    return n
