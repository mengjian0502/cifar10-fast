from qtorch import BlockFloatingPoint, FixedPoint, FloatingPoint
from qtorch.quant import quantizer, Quantizer
from core import *
from torch_backend import *
from td import Conv2d_TD, Linear_TD, Conv2d_col_TD

# Network definition
def conv_bn_TD(c_in, c_out, quant, gamma=0.0, alpha=0.0, block_size=16):
    IBM_half = FloatingPoint(exp=6, man=9)
    quant_half = Quantizer(IBM_half, IBM_half, "nearest", "nearest")
    return {
        'conv': Conv2d_TD(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False, gamma=gamma, alpha=alpha, block_size=block_size), 
        'quant1': quant(),
        'bn': BatchNorm(c_out), 
        'relu': nn.ReLU(True),
        'quant2': quant()
    }

def conv_bn(c_in, c_out, quant):
    IBM_half = FloatingPoint(exp=6, man=9)
    quant_half = Quantizer(IBM_half, IBM_half, "nearest", "nearest")
    return {
        'half_quant': quant_half(),
        'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False), 
        'quant': quant(),
        'bn': BatchNorm(c_out), 
        'relu': nn.ReLU(True),
    }   

def residual(c, quant):
    return {
        'in': Identity(),
        'res1': conv_bn_TD(c, c),
        'res2': conv_bn_TD(c, c),
        'add': (Add(), ['in', 'res2/relu']),
        'quant': quant()
    }

def net_low(channels=None, weight=0.125, pool=nn.MaxPool2d(2), extra_layers=(), res_layers=('layer1', 'layer3'), gamma=0.0, alpha=0.0, block_size=16, quant=None):
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
        n[layer]['residual'] = residual(channels[layer], gamma=gamma, alpha=alpha, block_size=block_size, quant=quant)
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer], gamma=gamma, alpha=alpha, block_size=block_size, quant=quant)       
    return n
