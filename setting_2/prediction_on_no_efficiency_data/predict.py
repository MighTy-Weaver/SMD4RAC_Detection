import argparse
import os
import warnings

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from filter_data import AC_sparse_separate_dataset
from model import NN_regressor
from model import Transformer_encoder
from model import simple_LSTM_encoder

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=['lstm', 'bilstm', 'transformer', 'lstm-transformer', 'bilstm-transformer'],
                    default='lstm-transformer')
parser.add_argument("--lr", help="learning rate", default=5e-5, type=float)
parser.add_argument("--epoch", help="epochs", default=100, type=int)
parser.add_argument("--bs", help="batch size", default=64, type=int)
parser.add_argument("--data_mode", help="use sparse data or daily data", choices=['daily', 'sparse'], default='sparse',
                    type=str)
parser.add_argument("--room", default=1, type=float, help="Room ratio for sampling rooms")
parser.add_argument("--ratio", default=0.8, type=float, help="train data ratio")

parser.add_argument("--data", default=2000, type=int, help="The number of data to be trained")
parser.add_argument("--gs", help="group size for sparse dataset", default=144, type=int)
parser.add_argument("--gpu", help="gpu number", default=1, type=int)

parser.add_argument("--test", help="run in test mode", default=0, type=int)

args = parser.parse_args()

# Check device status
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('CUDA available:', torch.cuda.is_available())
print(torch.cuda.get_device_name())
print('Device number:', torch.cuda.device_count())
print(torch.cuda.get_device_properties(device))

if args.test == 1:
    warnings.warn("Running in TEST mode!")

# Set the GPU device
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(args.gpu)

# Build the model
if args.model == 'lstm':
    encoder = simple_LSTM_encoder(bidirectional=False, feature_num=12).to(device)
elif args.model == 'bilstm':
    encoder = simple_LSTM_encoder(bidirectional=True, feature_num=12).to(device)
elif args.model == 'transformer':
    encoder = Transformer_encoder(gs=args.gs, feature_num=12, num_head=6, mode='flat').to(device)
elif args.model == 'lstm-transformer':
    encoder = Transformer_encoder(gs=args.gs, feature_num=12, num_head=6, mode='lstm', bidirectional=False).to(device)
elif args.model == 'bilstm-transformer':
    encoder = Transformer_encoder(gs=args.gs, feature_num=12, num_head=6, mode='lstm', bidirectional=True).to(device)
else:
    raise NotImplementedError(
        "Model type other than 'lstm' or 'attn lstm' or 'transformer' hasnot been implemented yet")

model = NN_regressor(output_dimension=2, encoder=encoder).to(device)
model.load_state_dict(torch.load("/home/wwangbw/projects/AC-Detection-Validation/setting_1/best.pth"))

# Training settings
num_epoch = args.epoch
batch_size = args.bs
learning_rate = args.lr
data_mode = args.data_mode
group_size = args.gs

training_dataset = AC_sparse_separate_dataset('trn', test=args.test == 1, group_size=group_size, trn_ratio=args.ratio,
                                              cla=True, total_number=args.data)
train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

model.eval()
trn_total_pred, trn_total_room = [], []
for inputs, rooms in tqdm(train_loader):
    inputs = inputs.to(device)
    labels = rooms.type(torch.int64).to(device)
    outputs = model(inputs)

    predicted_answer = torch.argmax(outputs, dim=-1)
    truth_answer = rooms.detach().cpu()
    trn_total_pred.extend(predicted_answer.tolist())
    trn_total_room.extend(truth_answer.tolist())

print(trn_total_pred)
print(trn_total_room)
