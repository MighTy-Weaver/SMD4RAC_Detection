import argparse
import os
import warnings

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from dataloader import AC_sparse_separate_dataset
from model import NN_regressor
from model import Transformer_encoder
from model import simple_LSTM_encoder

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=['lstm', 'bilstm', 'transformer', 'lstm-transformer', 'bilstm-transformer'],
                    default='lstm')
parser.add_argument("--lr", help="learning rate", default=5e-5, type=float)
parser.add_argument("--epoch", help="epochs", default=100, type=int)
parser.add_argument("--bs", help="batch size", default=64, type=int)
parser.add_argument("--data_mode", help="use sparse data or daily data", choices=['daily', 'sparse'], default='sparse',
                    type=str)
parser.add_argument("--room", default=1, type=float, help="Room ratio for sampling rooms")
parser.add_argument("--ratio", default=0.8, type=float, help="train data ratio")

parser.add_argument("--data", default=100000, type=int, help="The number of data to be trained")
parser.add_argument("--gs", help="group size for sparse dataset", default=25, type=int)
parser.add_argument("--gpu", help="gpu number", default=2, type=int)

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

# Training settings
num_epoch = args.epoch
batch_size = args.bs
learning_rate = args.lr
data_mode = args.data_mode
group_size = args.gs
criterion = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
save_path = './ckpt/{}_checkpoint_bs{}_e{}_lr{}_mode{}_gs{}_rat{}_roomrat{}_numdata{}/'.format(args.model, batch_size,
                                                                                               num_epoch,
                                                                                               learning_rate,
                                                                                               data_mode, group_size,
                                                                                               args.ratio, args.room,
                                                                                               args.data)

# Make checkpoint save path
if not os.path.exists('./ckpt/'):
    os.mkdir('./ckpt/')
if not os.path.exists(save_path):
    os.mkdir(save_path)

training_dataset = AC_sparse_separate_dataset('trn', test=args.test == 1, group_size=group_size, trn_ratio=args.ratio,
                                              cla=True, total_number=args.data)
validation_dataset = AC_sparse_separate_dataset('val', test=args.test == 1, group_size=group_size, trn_ratio=args.ratio,
                                                cla=True, total_number=args.data)
train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

lr_scheduler = get_scheduler(name='linear', optimizer=optimizer,
                             num_warmup_steps=0, num_training_steps=num_epoch * len(train_loader))

progress_bar = tqdm(range(num_epoch * len(train_loader)))

record = {i: [] for i in ['trn_f1', 'val_f1', 'trn_loss', 'val_loss', 'trn_acc', 'val_acc']}
# Start training
for epoch in range(num_epoch):
    epoch_loss = 0

    model.train()
    trn_total_pred, trn_total_label = [], []
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.type(torch.int64).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        epoch_loss += loss.item()

        predicted_answer = torch.argmax(outputs, dim=-1)
        truth_answer = labels.detach().cpu()
        trn_total_pred.extend(predicted_answer.tolist())
        trn_total_label.extend(truth_answer.tolist())
        progress_bar.update(1)
    trn_f1 = f1_score(trn_total_label, trn_total_pred)
    trn_acc = accuracy_score(trn_total_label, trn_total_pred)

    model.eval()
    val_total_pred, val_total_label = [], []
    with torch.no_grad():
        val_epoch_loss = 0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.type(torch.int64).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_epoch_loss += loss.item()

            predicted_answer = torch.argmax(outputs, dim=-1)
            truth_answer = labels.detach().cpu()
            val_total_pred.extend(predicted_answer.tolist())
            val_total_label.extend(truth_answer.tolist())
        val_f1 = f1_score(val_total_label, val_total_pred)
        val_acc = accuracy_score(val_total_label, val_total_pred)
        print("Training Epoch {}\tTraining Loss {}\tValidation Loss {}".format(epoch + 1,
                                                                               epoch_loss / len(train_loader),
                                                                               val_epoch_loss / len(val_loader)))
        print('Training Acc = {}\tValidation Acc = {}'.format(round(trn_acc, 4), round(val_acc, 4)))
        print('Training F1 = {}\tValidation F1 = {}'.format(round(trn_f1, 4), round(val_f1, 4)))

        record['trn_loss'].append(epoch_loss / len(train_loader))
        record['val_loss'].append(val_epoch_loss / len(val_loader))
        record['trn_acc'].append(trn_acc)
        record['val_acc'].append(val_acc)
        record['trn_f1'].append(trn_f1)
        record['val_f1'].append(val_f1)
        print("MAX f1 is {} for train and {} for val".format(max(record['trn_f1']), max(record['val_f1'])))
        print("MAX acc is {} for train and {} for val".format(max(record['trn_acc']), max(record['val_acc'])))

    if val_acc >= max(record['val_acc']):
        np.save(os.path.join(save_path, 'best_pred.npy'), trn_total_pred + val_total_pred)
        np.save(os.path.join(save_path, 'best_label.npy'), trn_total_label + val_total_label)
        np.save(os.path.join(save_path, 'best_train_label.npy'), trn_total_label)
        np.save(os.path.join(save_path, 'best_valid_label.npy'), val_total_label)
        np.save(os.path.join(save_path, 'best_train_pred.npy'), trn_total_pred)
        np.save(os.path.join(save_path, 'best_valid_pred.npy'), val_total_pred)
        torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'.format(epoch + 1)))
    np.save(os.path.join(save_path, 'record.npy'), record)

