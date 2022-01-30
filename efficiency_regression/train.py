import argparse
import os
import warnings

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import trange

from dataloader import AC_Normal_Dataset
from model import LSTM_encoder
from model import NN_regressor

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="learning rate", default=0.00045, type=float)
parser.add_argument("--epoch", help="epochs", default=50, type=int)
parser.add_argument("--bs", help="batch size", default=256, type=int)
parser.add_argument("--gpu", help="gpu number", default=3, type=int)
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
encoder = LSTM_encoder(bidirectional=True).to(device)
model = NN_regressor(output_dimension=1, encoder=encoder).to(device)

# Training settings
num_epoch = args.epoch
batch_size = args.bs
learning_rate = args.lr
criterion = MSELoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
save_path = './model_checkpoint/'
save_step = 2

# Make checkpoint save path
if not os.path.exists(save_path):
    os.mkdir(save_path)

training_dataset = AC_Normal_Dataset('trn', test=args.test == 1)
validation_dataset = AC_Normal_Dataset('val', test=args.test == 1)
train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

record = {i: [] for i in ['trn_loss', 'trn_acc', 'trn_f1', 'val_loss', 'val_acc', 'val_f1']}
# Start training
for epoch in trange(num_epoch, desc="Epoch: "):
    epoch_loss = 0

    model.train()
    trn_total, trn_correct = 0, 0
    trn_total_pred, trn_total_label = [], []
    for inputs, labels in tqdm(train_loader):
        trn_total += len(inputs)
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        predicted_answer = torch.argmax(outputs, dim=1)
        truth_answer = labels.detach().cpu()
        trn_total_pred.extend(predicted_answer.tolist())
        trn_total_label.extend(truth_answer.tolist())
        trn_correct += sum([predicted_answer[ind] == truth_answer[ind] for ind in range(len(predicted_answer))])

    model.eval()
    val_total_pred, val_total_label = [], []
    with torch.no_grad():
        val_epoch_loss = 0
        val_total, val_correct = 0, 0
        for inputs, labels in val_loader:
            val_total += len(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_epoch_loss += loss.item()

            predicted_answer = torch.argmax(outputs, dim=1)
            truth_answer = labels.detach().cpu()
            val_total_pred.extend(predicted_answer.tolist())
            val_total_label.extend(truth_answer.tolist())
            val_correct += sum(
                predicted_answer[ind] == truth_answer[ind]
                for ind in range(len(predicted_answer))
            )

        print("Training Epoch {}\tTraining Loss {}\tValidation Loss {}".format(epoch + 1,
                                                                               epoch_loss / len(train_loader),
                                                                               val_epoch_loss / len(val_loader)))
        print('Training Accuracy = {}\tValidation Accuracy = {}'.format(round(int(trn_correct) / int(trn_total), 3),
                                                                        round(int(val_correct) / int(val_total), 3)))
        print("Training F1 score = {}\nValidation F1 score = {}".format(
            round(f1_score(trn_total_label, trn_total_pred), 3), round(f1_score(val_total_label, val_total_pred), 3)))

        record['trn_loss'].append(epoch_loss / len(train_loader))
        record['val_loss'].append(val_epoch_loss / len(val_loader))
        record['trn_acc'].append(round(int(trn_correct) / int(trn_total), 3))
        record['val_acc'].append(round(int(val_correct) / int(val_total), 3))
        record['trn_f1'].append(round(f1_score(trn_total_label, trn_total_pred), 3))
        record['val_f1'].append(round(f1_score(val_total_label, val_total_pred), 3))

    if (epoch + 1) % save_step == 0:
        torch.save(model.state_dict(), os.path.join(save_path, 'epoch{}.pth'.format(epoch + 1)))
        np.save(os.path.join(save_path, 'epoch{}_record.npy'.format(epoch + 1)), record)
