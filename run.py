import os
import argparse
from models.FCNs import FCN8s, FCN16s, FCN32s
from models.UNet import UNet
from models.UNetplus import NestedUNet
from models.CENet import CE_Net
from models.CPFNet import CPFNet
from train import train
from test import test

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='FCN8s', help='The model to use')
parser.add_argument('--train', action='store_true', help='Train the model')
args = parser.parse_args()

if args.model not in ['FCN8s', 'FCN16s', 'FCN32s', 'UNet', 'UNet++', 'CENet', 'CPFNet']:
    print(args.model + ' model is not supported!')
    exit()
else:
    if args.model == 'FCN8s':
        model = FCN8s()
    elif args.model == 'FCN16s':
        model = FCN16s()
    elif args.model == 'FCN32s':
        model = FCN32s()
    elif args.model == 'UNet':
        model = UNet()
    elif args.model == 'UNet++':
        model = NestedUNet()
    elif args.model == 'CENet':
        model = CE_Net()
    else:
        model = CPFNet()

model_dir = 'saved_model/'
if args.train:
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    train(model, model_path=model_dir + 'model_' + args.model + '.pth')
else:
    # default: test
    test(model, model_path=model_dir + 'model_' + args.model + '.pth', res_dir='predict/' + args.model + '/')
