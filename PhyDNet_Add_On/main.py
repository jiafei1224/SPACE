import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
import time
from models.models import ConvLSTM, PhyCell, EncoderRNN
# from data.moving_mnist import MovingMNIST
from data.dataloader import Data
from constrain_moments import K2M
from skimage.metrics import structural_similarity as ssim
import argparse
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from torchvision.utils import save_image
import os

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', required=True)
parser.add_argument('--print_every', type=int, default=1, help='')
parser.add_argument('--eval_every', type=int, default=1, help='')
args = parser.parse_args()
with open(args.config_file, "r") as setting:
    cfg = yaml.safe_load(setting)

# load config
task_type = cfg['task_type']
train_frame_path = cfg['train_frame_path']
val_frame_path = cfg['val_frame_path']
# mask_path = cfg['mask_path']
# train_label_path = cfg['train_label_path']
# val_label_path = cfg['val_label_path']
# test_label_path = cfg['test_label_path']
# save_path = cfg['save_path']
save_frames_every = cfg['save_frames_every']
num_epoch = cfg['num_epoch']
batch_size = cfg['batch_size']
# teacher_forcing_prob = cfg['teacher_forcing_prob']
first_n_frame_dynamics = cfg['first_n_frame_dynamics']
frame_interval = cfg['frame_interval']
learning_rate = cfg['learning_rate']
save_name = cfg['save_name']
max_frames = cfg['max_frames']
load_model_path = cfg['load_model_path']
print(load_model_path)
print(num_epoch)
print(train_frame_path)
print(val_frame_path)


# parser.add_argument('--root', type=str, default='data/')
# parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
# parser.add_argument('--nepochs', type=int, default=2001, help='nb of epochs')
# parser.add_argument('--save_name', type=str, default='phydnet', help='')
# args = parser.parse_args()


# mm = MovingMNIST(root=args.root, is_train=True, n_frames_input=10, n_frames_output=10, num_objects=[2])
# train_loader = torch.utils.data.DataLoader(dataset=mm, batch_size=args.batch_size, shuffle=True, num_workers=0)

# mm = MovingMNIST(root=args.root, is_train=False, n_frames_input=10, n_frames_output=10, num_objects=[2])
# test_loader = torch.utils.data.DataLoader(dataset=mm, batch_size=args.batch_size, shuffle=False, num_workers=0)

train_dataset = Data(train_frame_path, frame_interval, first_n_frame_dynamics, max_frames)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

val_dataset = Data(val_frame_path, frame_interval, first_n_frame_dynamics, max_frames)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

constraints = torch.zeros((49, 7, 7)).to(device)
ind = 0
for i in range(0, 7):
    for j in range(0, 7):
        constraints[ind, i, j] = 1
        ind += 1


def trainIters(encoder, num_epoch, learning_rate, print_every=10, eval_every=10, name=''):
    train_losses = []
    best_mse = float('inf')

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    scheduler_enc = ReduceLROnPlateau(encoder_optimizer, mode='min', patience=2, factor=0.1, verbose=True)
    criterion = nn.MSELoss()

    train_stats_file = open('./save51/train_stats.txt', 'a')
    val_stats_file = open('./save51/val_stats.txt', 'a')

    for epoch in range(0, num_epoch):
        t0 = time.time()
        loss_epoch = 0
        teacher_forcing_ratio = np.maximum(0 , 1 - epoch * 0.003)

        print("Total train iterations: {}".format(len(train_loader)))
        for i, out in tqdm(enumerate(train_loader, 0)):
            input_tensor = out[1].to(device)
            target_tensor = out[2].to(device)
            loss, output_sequence, gt_sequence = train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer,
                                                                criterion, teacher_forcing_ratio)
            loss_epoch += loss

        # save frames
        os.makedirs('./save51/frames/train/epoch_{}/gen/'.format(epoch), exist_ok=True)
        os.makedirs('./save51/frames/train/epoch_{}/real/'.format(epoch), exist_ok=True)
        for i in range(len(output_sequence)):
            save_image(output_sequence[i][0], './save51/frames/train/epoch_{}/gen/{}.png'.format(epoch, i))
            save_image(gt_sequence[i][0], './save51/frames/train/epoch_{}/real/{}.png'.format(epoch, i))

        train_losses.append(loss_epoch)
        train_stats_file.write(str(loss_epoch) + '\n')
        if (epoch + 1) % print_every == 0:
            print('epoch ', epoch, ' loss ', loss_epoch, ' time epoch ', time.time() - t0)

        if (epoch + 1) % eval_every == 0:
            mse, mae, ssim = evaluate(encoder, val_loader, epoch, val_stats_file)
            scheduler_enc.step(mse)
            torch.save(encoder.state_dict(), 'save51/encoder_{}.pth'.format(name))

    train_stats_file.close()
    val_stats_file.close()

    return train_losses


def train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion, teacher_forcing_ratio):
    encoder_optimizer.zero_grad()
    # input_tensor : torch.Size([batch_size, input_length, channels, cols, rows])
    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)
    loss = 0
    output_sequence = []
    gt_sequence = []
    for ei in range(input_length - 1):
        encoder_output, encoder_hidden, output_image, _, _ = encoder(input_tensor[:, ei, :, :, :], (ei == 0))
        loss += criterion(output_image, input_tensor[:, ei + 1, :, :, :])
        output_sequence.append(input_tensor[:, ei, :, :, :].detach().cpu())
        gt_sequence.append(input_tensor[:, ei, :, :, :].detach().cpu())

    decoder_input = input_tensor[:, -1, :, :, :]  # first decoder input = last image of input sequence
    output_sequence.append(decoder_input.detach().cpu())
    gt_sequence.append(decoder_input.detach().cpu())

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    for di in range(target_length):
        decoder_output, decoder_hidden, output_image, _, _ = encoder(decoder_input)
        target = target_tensor[:, di, :, :, :]
        loss += criterion(output_image, target)
        if use_teacher_forcing:
            decoder_input = target  # Teacher forcing
        else:
            decoder_input = output_image
        output_sequence.append(output_image.detach().cpu())
        gt_sequence.append(target.detach().cpu())

    # Moment regularization  # encoder.phycell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
    k2m = K2M([7, 7]).to(device)
    for b in range(0, encoder.phycell.cell_list[0].input_dim):
        filters = encoder.phycell.cell_list[0].F.conv1.weight[:, b, :, :]  # (nb_filters,7,7)
        m = k2m(filters.double())
        m = m.float()
        loss += criterion(m, constraints)  # constrains is a precomputed matrix
    loss.backward()
    encoder_optimizer.step()
    return loss.item() / target_length, output_sequence, gt_sequence


def evaluate(encoder, loader, epoch, val_stats_file):
    total_mse, total_mae, total_ssim, total_bce = 0, 0, 0, 0
    t0 = time.time()
    # choices = [i for i in range(len(loader))]
    # choice = random.choice(choices)
    with torch.no_grad():
        for i, out in enumerate(loader, 0):
            output_sequence = []
            gt_sequence = []
            input_tensor = out[1].to(device)
            target_tensor = out[2].to(device)
            input_length = input_tensor.size()[1]
            target_length = target_tensor.size()[1]

            for ei in range(input_length - 1):
                encoder_output, encoder_hidden, _, _, _ = encoder(input_tensor[:, ei, :, :, :], (ei == 0))
                output_sequence.append(input_tensor[:, ei, :, :, :].detach().cpu())
                gt_sequence.append(input_tensor[:, ei, :, :, :].detach().cpu())

            decoder_input = input_tensor[:, -1, :, :, :]  # first decoder input= last image of input sequence
            output_sequence.append(decoder_input.detach().cpu())
            gt_sequence.append(decoder_input.detach().cpu())
            predictions = []

            for di in range(target_length):
                decoder_output, decoder_hidden, output_image, _, _ = encoder(decoder_input, False, False)
                decoder_input = output_image
                predictions.append(output_image.cpu())
                output_sequence.append(output_image.detach().cpu())
                target = target_tensor[:, di, :, :, :]
                gt_sequence.append(target.detach().cpu())

            input = input_tensor.cpu().numpy()
            target = target_tensor.cpu().numpy()
            predictions = np.stack(predictions)  # (10, batch_size, 1, 64, 64)
            predictions = predictions.swapaxes(0, 1)  # (batch_size,10, 1, 64, 64)

            mse_batch = np.mean((predictions - target) ** 2, axis=(0, 1, 2)).sum()
            mae_batch = np.mean(np.abs(predictions - target), axis=(0, 1, 2)).sum()
            total_mse += mse_batch
            total_mae += mae_batch

            for a in range(0, target.shape[0]):
                for b in range(0, target.shape[1]):
                    total_ssim += ssim(target[a, b, 0,], predictions[a, b, 0,]) / (target.shape[0] * target.shape[1])

            cross_entropy = -target * np.log(predictions) - (1 - target) * np.log(1 - predictions)
            cross_entropy = cross_entropy.sum()
            cross_entropy = cross_entropy / (batch_size * target_length)
            total_bce += cross_entropy


        os.makedirs('./save51/frames/val/epoch_{}/gen/'.format(epoch), exist_ok=True)
        os.makedirs('./save51/frames/val/epoch_{}/real/'.format(epoch), exist_ok=True)
        for j in range(len(output_sequence)):
            save_image(output_sequence[j][0], './save51/frames/val/epoch_{}/gen/{}.png'.format(epoch, j))
            save_image(gt_sequence[j][0], './save51/frames/val/epoch_{}/real/{}.png'.format(epoch, j))

    val_stats_file.write(
        str(total_mse / len(loader)) + '\n')
    print('eval mse ', total_mse / len(loader), ' eval mae ', total_mae / len(loader), ' eval ssim ',
          total_ssim / len(loader), ' time= ', time.time() - t0)
    return total_mse / len(loader), total_mae / len(loader), total_ssim / len(loader)


phycell = PhyCell(input_shape=(32, 32), input_dim=64, F_hidden_dims=[49], n_layers=1, kernel_size=(7, 7), device=device)
convcell = ConvLSTM(input_shape=(32, 32), input_dim=64, hidden_dims=[128, 128, 64], n_layers=3, kernel_size=(3, 3),
                    device=device)
encoder = EncoderRNN(phycell, convcell, device)

#Loadmodel
if load_model_path is not None:
    encoder.load_state_dict(torch.load(load_model_path))
    print('Loaded pre-trained model from ' + str(load_model_path) + ' successfully.')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('phycell ', count_parameters(phycell))
print('convcell ', count_parameters(convcell))
print('encoder ', count_parameters(encoder))

trainIters(encoder, num_epoch, learning_rate, print_every=args.print_every, eval_every=args.eval_every, name=save_name)

# encoder.load_state_dict(torch.load('save/encoder_phydnet.pth'))
# encoder.eval()
# mse, mae,ssim = evaluate(encoder,test_loader)
