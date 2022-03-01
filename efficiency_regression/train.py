import argparse
import os
import warnings

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import trange

from dataloader import AC_Normal_Dataset
from dataloader import AC_Sparse_Dataset
from model import NN_regressor
from model import simple_LSTM_encoder

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="learning rate", default=0.00045, type=float)
parser.add_argument("--epoch", help="epochs", default=50, type=int)
parser.add_argument("--bs", help="batch size", default=256, type=int)
parser.add_argument("--gpu", help="gpu number", default=3, type=int)
parser.add_argument("--test", help="run in test mode", default=0, type=int)
parser.add_argument("--data_mode", help="use sparse data or daily data", choices=['daily', 'sparse'], type=str)
parser.add_argument("--gs", help="group size for sparse dataset", default=200, type=int)
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
encoder = simple_LSTM_encoder(bidirectional=True, feature_num=12).to(device)
model = NN_regressor(output_dimension=1, encoder=encoder).to(device)

# Training settings
num_epoch = args.epoch
batch_size = args.bs
learning_rate = args.lr
data_mode = args.data_mode
group_size = args.gs
criterion = MSELoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
save_path = './LSTM_model_checkpoint_bs{}_e{}_lr{}_mode{}_gs{}/'.format(batch_size, num_epoch, learning_rate, data_mode,
                                                                        group_size)
save_step = 2

# Make checkpoint save path
if not os.path.exists(save_path):
    os.mkdir(save_path)

if data_mode == 'daily':
    training_dataset = AC_Normal_Dataset('trn', test=args.test == 1)
    validation_dataset = AC_Normal_Dataset('val', test=args.test == 1)
else:
    training_dataset = AC_Sparse_Dataset('trn', test=args.test == 1, group_size=group_size)
    validation_dataset = AC_Sparse_Dataset('val', test=args.test == 1, group_size=group_size)
train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

record = {i: [] for i in ['trn_r2', 'val_r2', 'trn_loss', 'val_loss']}
# Start training
for epoch in trange(num_epoch, desc="Epoch: "):
    epoch_loss = 0

    model.train()
    trn_total_pred, trn_total_label = [], []
    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.type(torch.float32).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels.reshape(-1, 1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        predicted_answer = outputs
        truth_answer = labels.detach().cpu()
        trn_total_pred.extend(predicted_answer.tolist())
        trn_total_label.extend(truth_answer.tolist())
    trn_r2 = r2_score(trn_total_label, trn_total_pred)

    model.eval()
    val_total_pred, val_total_label = [], []
    with torch.no_grad():
        val_epoch_loss = 0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.reshape(-1, 1))
            val_epoch_loss += loss.item()

            predicted_answer = torch.argmax(outputs, dim=1)
            truth_answer = labels.detach().cpu()
            val_total_pred.extend(predicted_answer.tolist())
            val_total_label.extend(truth_answer.tolist())
        val_r2 = r2_score(val_total_label, val_total_pred)
        print("Training Epoch {}\tTraining Loss {}\tValidation Loss {}".format(epoch + 1,
                                                                               epoch_loss / len(train_loader),
                                                                               val_epoch_loss / len(val_loader)))
        print('Training R2 = {}\tValidation R2 = {}'.format(round(trn_r2, 4), round(val_r2, 4)))

        record['trn_loss'].append(epoch_loss / len(train_loader))
        record['val_loss'].append(val_epoch_loss / len(val_loader))
        record['trn_r2'].append(trn_r2)
        record['val_r2'].append(val_r2)

    if (epoch + 1) % save_step == 0:
        torch.save(model.state_dict(), os.path.join(save_path, 'epoch{}.pth'.format(epoch + 1)))
        np.save(os.path.join(save_path, 'epoch{}_record.npy'.format(epoch + 1)), record)
