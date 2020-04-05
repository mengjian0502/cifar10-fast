"""
Customized ResNet-9 Fast training with CIFAR10 dataset
"""
import argparse
import os
import utils
import numpy as np

from core import *
from torch_backend import *
from model import *
from td import Conv2d_TD, Linear_TD, Conv2d_col_TD

from qtorch import BlockFloatingPoint, FixedPoint, FloatingPoint
from qtorch.quant import quantizer, Quantizer

num_types = ["weight", "activate", "grad", "error", "momentum", "acc"]

parser = argparse.ArgumentParser(description='resnet9 fast CIFAR10 training + Targeted Dropout')
parser.add_argument('--epochs', type=int, default=24, 
                    help='number of epochs to train (default: 24)')
parser.add_argument('--save_file', type=str, default=None,
                    help='path to save file for model')
parser.add_argument('--TD_gamma', type=float, default=0.0,
                    help='gamma value for targeted dropout')
parser.add_argument('--TD_alpha', type=float, default=0.0,
                    help='alpha value for targeted dropout')
parser.add_argument('--block_size', type=int, default=16,
                    help='block size for dropout')
parser.add_argument('--evaluate', type=str, default=None,
                    help='model file for accuracy evaluation')
parser.add_argument('--TD_gamma_final', type=float, default=-1.0,
                    help='final gamma value for targeted dropout')
parser.add_argument('--TD_alpha_final', type=float, default=-1.0,
                    help='final alpha value for targeted dropout')
parser.add_argument('--ramping_power', type=float, default=3.0,
                    help='power of ramping schedule')
parser.add_argument('--LP', action='store_true', help='IBM FP8')

for num in num_types:
    parser.add_argument('--{}-man'.format(num), type=int, default=-1, metavar='N',
                        help='number of bits to use for mantissa of {}; -1 if full precision.'.format(num))
    parser.add_argument('--{}-exp'.format(num), type=int, default=-1, metavar='N',
                        help='number of bits to use for exponent of {}; -1 if full precision.'.format(num))
    parser.add_argument('--{}-rounding'.format(num), type=str, default='stochastic', metavar='S',
                        choices=["stochastic","nearest"],
                        help='rounding method for {}, stochastic or nearest'.format(num))

args = parser.parse_args()


def main():
    DATA_DIR = './data'
    dataset = cifar10(root=DATA_DIR)
    timer = Timer()
    print('Preprocessing training data')
    transforms = [
        partial(normalise, mean=np.array(cifar10_mean, dtype=np.float32), std=np.array(cifar10_std, dtype=np.float32)),
        partial(transpose, source='NHWC', target='NCHW'), 
    ]
    train_set = list(zip(*preprocess(dataset['train'], [partial(pad, border=4)] + transforms).values()))
    print(f'Finished in {timer():.2} seconds')
    print('Preprocessing test data')
    valid_set = list(zip(*preprocess(dataset['valid'], transforms).values()))
    print(f'Finished in {timer():.2} seconds')

    epochs=args.epochs
    lr_schedule = PiecewiseLinear([0, 5, epochs], [0, 0.4, 0])
    batch_size = 512
    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]
    N_runs = 1

    train_batches = DataLoader(Transform(train_set, train_transforms), batch_size, shuffle=True, set_random_choices=True, drop_last=True)
    valid_batches = DataLoader(valid_set, batch_size, shuffle=False, drop_last=False)
    lr = lambda step: lr_schedule(step/len(train_batches))/batch_size

    ##### Fetch the LP attributes ########
    if args.LP:
        quantizers = {}
        for num in num_types:
            num_rounding = getattr(args, "{}_rounding".format(num))
            num_man = getattr(args, "{}_man".format(num))
            num_exp = getattr(args, "{}_exp".format(num))
            number = FloatingPoint(exp=num_exp, man=num_man)
            logger.info("{}: {} rounding, {}".format(num, num_rounding,
                                           number))
            quantizers[num] = quantizer(forward_number=number, forward_rounding=num_rounding)

    summaries = []
    for i in range(N_runs):
        print(f'Starting Run {i} at {localtime()}')
        model = Network(net(gamma=args.TD_gamma, alpha=args.TD_alpha, block_size=args.block_size)).to(device)

        Hooks_input = utils.add_input_record_Hook(model)
        opts = [SGD(trainable_params(model).values(), {'lr': lr, 'weight_decay': Const(5e-4*batch_size), 'momentum': Const(0.9)})]
        logs, state = Table(), {MODEL: model, LOSS: x_ent_loss, OPTS: opts}
        
        activation_sparsity=0.0
        asparse = []
        wsparse = []
        for epoch in range(epochs):
            td_gamma, td_alpha = update_gamma_alpha(epoch, model)
            
            weight_sparsity = utils.get_weight_sparsity(model)
            logs.append(union({'epoch': epoch+1}, {'lr': lr_schedule(epoch+1)}, {'gamma': td_gamma}, {'alpha': td_alpha}, {'wspar': weight_sparsity}, {'aspar': round(activation_sparsity,4)}, train_epoch(state, Timer(torch.cuda.synchronize), train_batches, valid_batches)))
            activation_sparsity = utils.get_activation_sparsity(Hooks_input).item()

            asparse.append(activation_sparsity)
            wsparse.append(weight_sparsity)
        
        print(f'Avg weight sparsity: {np.mean(wsparse)} | Avg act sparsity: {np.mean(asparse)}')

    logs.df().query(f'epoch=={epochs}')[['train_acc', 'valid_acc']].describe()

    if args.save_file is not None:
        torch.save({
            'state_dict': model.state_dict(),
            'args': args}, 
            os.path.join('checkpoint', args.save_file))

def update_gamma_alpha(epoch, model):
    if args.TD_gamma_final > 0:
        TD_gamma = args.TD_gamma_final - (((args.epochs - 1 - epoch)/(args.epochs - 1)) ** args.ramping_power) * (args.TD_gamma_final - args.TD_gamma)
        for m in model.modules():
            if hasattr(m, 'gamma'):
                m.gamma = TD_gamma
    else:
        TD_gamma = args.TD_gamma
    if args.TD_alpha_final > 0:
        TD_alpha = args.TD_alpha_final - (((args.epochs - 1 - epoch)/(args.epochs - 1)) ** args.ramping_power) * (args.TD_alpha_final - args.TD_alpha)
        for m in model.modules():
            if hasattr(m, 'alpha'):
                m.alpha = TD_alpha
    else:
        TD_alpha = args.TD_alpha
    return TD_gamma, TD_alpha

if __name__ == '__main__':
    main()
