import argparse
import os

import torch
from torch.nn import CrossEntropyLoss
from torch.nn import TripletMarginWithDistanceLoss
from torch.optim import Adam
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import trange

from dataloader import AC_Normal_Dataset
from dataloader import AC_Triplet_Dataset
from model import LSTM_encoder
from model import NN
from utils import cosine_similarity_loss

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--encoder_lr", help="learning rate", default=0.00045, type=float)
parser.add_argument("--classifier_lr", help="learning rate", default=0.0006, type=float)
parser.add_argument("--epoch", help="epochs", default=50, type=int)
parser.add_argument("--encoder_bs", help="batch size", default=786, type=int)
parser.add_argument("--classifier_bs", help="batch size", default=256, type=int)
parser.add_argument("--gpu", help="gpu number", default=2, type=int)
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
if not os.path.exists('./model_checkpoint_triplet'):
    os.mkdir('./model_checkpoint_triplet/')

# Build the model
encoder = LSTM_encoder(hidden_size=256, bidirectional=True).to(device)
classifier = NN(input_dimension=encoder.get_output_length(), output_dimension=2).to(device)

# Training settings
num_epoch = args.epoch
encoder_batch_size = args.encoder_bs
classifier_batch_size = args.classifier_bs
encoder_learning_rate = args.encoder_lr
classifier_learning_rate = args.classifier_lr
triplet_criterion = TripletMarginWithDistanceLoss(margin=0.2, distance_function=cosine_similarity_loss)
classifier_criterion = CrossEntropyLoss()
encoder_optimizer = AdamW(encoder.parameters(), lr=encoder_learning_rate)
classifier_optimizer = Adam(classifier.parameters(), lr=classifier_learning_rate)
save_path = './model_checkpoint_triplet/'
save_step = 1

triplet_dataset = AC_Triplet_Dataset()
training_dataset = AC_Normal_Dataset('trn')
validation_dataset = AC_Normal_Dataset('val')
triplet_loader = DataLoader(triplet_dataset, batch_size=encoder_batch_size, shuffle=True)
train_loader = DataLoader(training_dataset, batch_size=classifier_batch_size, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=classifier_batch_size, shuffle=True)

# Start training the encoder
for epoch in trange(num_epoch, desc="Training encoder:"):
    epoch_loss = 0

    encoder.train()
    for anchor, pos, neg in tqdm(triplet_loader):
        anchor = anchor.to(device)
        pos = pos.to(device)
        neg = neg.to(device)
        anchor_emb = encoder(anchor)
        pos_emb = encoder(pos)
        neg_emb = encoder(neg)
        loss = triplet_criterion(anchor_emb, pos_emb, neg_emb)

        encoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()

        epoch_loss += loss.item()

        print("Training Epoch {}\tTraining Loss {}".format(epoch + 1, epoch_loss / len(train_loader), ))

    if (epoch + 1) % save_step == 0:
        torch.save(encoder.state_dict(), os.path.join(save_path, 'encoder_epoch{}.pth'.format(epoch + 1)))

encoder.eval()
for epoch in trange(num_epoch, desc='Training classifier'):
    epoch_loss = 0

    classifier.train()
    trn_total, trn_correct = 0, 0
    for inputs, labels in tqdm(train_loader):
        trn_total += len(inputs)

        inputs = inputs.to(device)
        labels = labels.to(device)
        encoding = encoder(inputs)
        outputs = classifier(encoding)
        loss = classifier_criterion(outputs, labels)

        classifier_optimizer.zero_grad()
        loss.backward()
        classifier_optimizer.step()

        epoch_loss += loss.item()

        predicted_answer = torch.argmax(outputs, dim=1)
        truth_answer = labels.detach().cpu()
        trn_correct += sum([predicted_answer[ind] == truth_answer[ind] for ind in range(len(predicted_answer))])

    classifier.eval()
    with torch.no_grad():
        val_epoch_loss = 0
        val_total, val_correct = 0, 0
        for inputs, labels in val_loader:
            val_total += len(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)
            encoding = encoder(inputs)
            outputs = classifier(encoding)
            loss = classifier_criterion(outputs, labels)
            val_epoch_loss += loss.item()

            predicted_answer = torch.argmax(outputs, dim=1)
            truth_answer = labels.detach().cpu()
            val_correct += sum(
                predicted_answer[ind] == truth_answer[ind]
                for ind in range(len(predicted_answer))
            )

        print("Training Epoch {}\tTraining Loss {}\tValidation Loss {}".format(epoch + 1,
                                                                               epoch_loss / len(train_loader),
                                                                               val_epoch_loss / len(val_loader)))
        print('Training Accuracy = {}\tValidation Accuracy = {}'.format(round(trn_correct / trn_total, 3),
                                                                        round(val_correct / val_total, 3)))
    if (epoch + 1) % save_step == 0:
        torch.save(classifier.state_dict(), os.path.join(save_path, 'classifier_epoch{}.pth'.format(epoch + 1)))
