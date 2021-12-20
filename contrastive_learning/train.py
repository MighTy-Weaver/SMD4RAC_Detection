import argparse
import os

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import trange

from dataloader import AC_Normal_Dataset
from model import LSTM_encoder

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="learning rate", default=0.00045, type=float)
parser.add_argument("--epoch", help="epochs", default=50, type=int)
parser.add_argument("--bs", help="batch size", default=128, type=int)
parser.add_argument("--gpu", help="gpu number", default=3, type=int)
args = parser.parse_args()

# Check device status
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('CUDA available:', torch.cuda.is_available())
print(torch.cuda.get_device_name())
print('Device number:', torch.cuda.device_count())
print(torch.cuda.get_device_properties(device))

# Set the GPU device
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(args.gpu)

# Make checkpoint save path
if not os.path.exists('./model_checkpoint'):
    os.mkdir('./model_checkpoint/')

# Build the model
model = LSTM_encoder(bidirectional=False).to(device)

# Training settings
num_epoch = args.epoch
batch_size = args.bs
learning_rate = args.lr
criterion = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
save_path = './model_checkpoint/'
save_step = 2

dataset = AC_Normal_Dataset()
train_loader = DataLoader(
    dataset[: int(len(dataset) * 0.9)], batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    dataset[int(len(dataset) * 0.9):], batch_size=batch_size, shuffle=True
)

# Start training
for epoch in trange(num_epoch):
    epoch_loss = 0

    model.train()
    for iter_id, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        writer.add_scalars('Loss_iter', {'train': loss.item()},
                           iter_id + epoch * len(train_IDLoader))

    model.eval()
    with torch.no_grad():
        val_epoch_loss = 0
        for inputs, labels in val_IDLoader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels -= 230
            outputs = model.val_forward(inputs)
            loss = criterion(outputs, labels)
            val_epoch_loss += loss.item()

        print("Training Epoch {}\tTraining Loss {}\tValidation Loss {}".format(epoch + 1,
                                                                               epoch_loss / len(train_IDLoader),
                                                                               val_epoch_loss / len(val_IDLoader)))
        writer.add_scalars('Loss_epoch',
                           {'train': epoch_loss / len(train_IDLoader), 'valid': val_epoch_loss / len(val_IDLoader)},
                           epoch + 1)

        if (epoch + 1) % Rk_step == 0:
            RT_gallery_input = val_RT.get_gallery_imgs().to(device)
            RT_gallery_output = model.encode_image(RT_gallery_input)
            Rk_dict = {}
            for inputs, labels in val_RTLoader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model.encode_image(inputs)
                for i in [1, 5, 10]:
                    Rk, _ = R_k(k=i, query_encoded=outputs, gallery_encoded=RT_gallery_output,
                                ground_truth_dict=val_RT.csv_dict)
                    Rk_dict['R{}'.format(i)] = Rk
            print("At epoch {}, R{} = {}, R{} = {}, R{} = {}".format(epoch + 1, 1, Rk_dict['R1'], 5, Rk_dict['R5'], 10,
                                                                     Rk_dict['R10']))
            writer.add_scalars('Rank_k_accuracy', Rk_dict, epoch + 1)

    if (epoch + 1) % save_step == 0:
        torch.save(model.state_dict(), os.path.join(save_path, 'epoch{}.pth'.format(epoch + 1)))
