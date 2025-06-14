# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from utils import create_output_dir, LossMeter, wrap
from wavenet_models import cross_entropy_loss, Encoder, ZDiscriminator
from wavenet import WaveNet
from data import DatasetSet
from tqdm import tqdm
from pathlib import Path
import numpy as np
from itertools import chain
import argparse
import os
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_

torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_start_method('spawn', force=True)


parser = argparse.ArgumentParser(
    description='PyTorch Code for A Universal Music Translation Network')
# Env options:
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 92)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--expName', type=str, required=True,
                    help='Experiment name')
parser.add_argument('--data',
                    metavar='D', type=Path, help='Data path', nargs='+')
parser.add_argument('--checkpoint', default='',
                    metavar='C', type=str, help='Checkpoint path')
parser.add_argument('--load-optimizer', action='store_true')
parser.add_argument('--per-epoch', action='store_true',
                    help='Save model per epoch')

# Distributed
parser.add_argument('--dist-url', default='env://',
                    help='Distributed training parameters URL')
parser.add_argument('--dist-backend', default='nccl')
parser.add_argument('--local-rank', type=int,
                    help='Ignored during training.')

# Data options
parser.add_argument('--seq-len', type=int, default=16000,
                    help='Sequence length')
parser.add_argument('--epoch-len', type=int, default=10000,
                    help='Samples per epoch')
parser.add_argument('--batch-size', type=int, default=32,
                    help='Batch size')
parser.add_argument('--num-workers', type=int, default=10,
                    help='DataLoader workers')
parser.add_argument('--data-aug', action='store_true',
                    help='Turns data aug on')
parser.add_argument('--magnitude', type=float, default=0.5,
                    help='Data augmentation magnitude.')
parser.add_argument('--encoder-lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--decoder-lr', type=float, default=1e-3,
                    help='Learning rate')
parser.add_argument('--discriminator-lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--lr-decay', type=float, default=0.98,
                    help='new LR = old LR * decay')
parser.add_argument('--short', action='store_true',
                    help='Run only a few batches per epoch for testing')
parser.add_argument('--h5-dataset-name', type=str, default='wav',
                    help='Dataset name in .h5 file')

# Encoder options
parser.add_argument('--latent-d', type=int, default=128,
                    help='Latent size')
parser.add_argument('--repeat-num', type=int, default=6,
                    help='No. of hidden layers')
parser.add_argument('--encoder-channels', type=int, default=128,
                    help='Hidden layer size')
parser.add_argument('--encoder-blocks', type=int, default=3,
                    help='No. of encoder blocks.')
parser.add_argument('--encoder-pool', type=int, default=800,
                    help='Number of encoder outputs to pool over.')
parser.add_argument('--encoder-final-kernel-size', type=int, default=1,
                    help='final conv kernel size')
parser.add_argument('--encoder-layers', type=int, default=10,
                    help='No. of layers in each encoder block.')
parser.add_argument('--encoder-func', type=str, default='relu',
                    help='Encoder activation func.')

# Decoder options
parser.add_argument('--blocks', type=int, default=4,
                    help='No. of wavenet blocks.')
parser.add_argument('--layers', type=int, default=10,
                    help='No. of layers in each block.')
parser.add_argument('--kernel-size', type=int, default=2,
                    help='Size of kernel.')
parser.add_argument('--residual-channels', type=int, default=128,
                    help='Residual channels to use.')
parser.add_argument('--skip-channels', type=int, default=128,
                    help='Skip channels to use.')

# Z discriminator options
parser.add_argument('--d-layers', type=int, default=3,
                    help='Number of 1d 1-kernel convolutions on the input Z vectors')
parser.add_argument('--d-channels', type=int, default=100,
                    help='1d convolutions channels')
parser.add_argument('--d-cond', type=int, default=1024,
                    help='WaveNet conditioning dimension')
parser.add_argument('--d-lambda', type=float, default=1e-2,
                    help='Adversarial loss weight.')
parser.add_argument('--p-dropout-discriminator', type=float, default=0.0,
                    help='Discriminator input dropout - if unspecified, no dropout applied')
parser.add_argument('--grad-clip', type=float,
                    help='If specified, clip gradients with specified magnitude')


class Trainer:
    def __init__(self, args):
        self.args = args
        self.args.n_datasets = len(self.args.data)
        self.expPath = Path('checkpoints') / args.expName

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        self.logger = create_output_dir(args, self.expPath)
        self.data = DatasetSet(args.data, args.seq_len, args)
        self.losses_recon = [
            LossMeter(f'recon {i}') for i in range(self.args.n_datasets)]
        self.loss_d_right = LossMeter('d')
        self.loss_total = LossMeter('total')

        self.evals_recon = [LossMeter(f'recon {i}')
                            for i in range(self.args.n_datasets)]
        self.eval_d_right = LossMeter('eval d')
        self.eval_total = LossMeter('eval total')

        self.encoder = Encoder(args)
        self.decoders = torch.nn.ModuleList(
            [WaveNet(args) for _ in range(self.args.n_datasets)])
        self.discriminator = ZDiscriminator(args)

        if args.checkpoint:
            checkpoint_args_path = os.path.dirname(
                args.checkpoint) + '/args.pth'
            checkpoint_args = torch.load(
                checkpoint_args_path, weights_only=False)

            self.start_epoch = checkpoint_args[-1] + 1
            states = torch.load(args.checkpoint)

            if 'musicnet' in args.checkpoint:
                self.encoder.load_state_dict(states['encoder_state'])
            else:
                self.encoder.load_state_dict(states['encoder_state'])
                for i, decoder in enumerate(self.decoders):
                    decoder.load_state_dict(states['decoder_state'][i])
                self.discriminator.load_state_dict(
                    states['discriminator_state'])

            self.logger.info('Loaded checkpoint parameters')
        else:
            self.start_epoch = 0

        self.encoder = torch.nn.parallel.DistributedDataParallel(
            self.encoder.to(args.rank), device_ids=[args.rank])
        self.decoders = torch.nn.ModuleList([torch.nn.parallel.DistributedDataParallel(
            decoder.to(args.rank), device_ids=[args.rank], find_unused_parameters=True) for decoder in self.decoders])
        self.discriminator = torch.nn.parallel.DistributedDataParallel(
            self.discriminator.to(args.rank), device_ids=[args.rank])
        self.logger.info('Created DistributedDataParallel')

        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(), lr=args.encoder_lr)
        self.decoder_optimizers = [optim.Adam(
            decoder.parameters(), lr=args.decoder_lr) for decoder in self.decoders]
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=args.discriminator_lr)

        if args.checkpoint and 'musicnet' not in args.checkpoint:
            self.encoder_optimizer.load_state_dict(
                states['encoder_optimizer_state'])
            for i, opt in enumerate(self.decoder_optimizers):
                opt.load_state_dict(states['decoder_optimizer_state'][i])
            self.d_optimizer.load_state_dict(
                states['d_optimizer_state'])

        self.encoder_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.encoder_optimizer, args.lr_decay)
        self.decoder_schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(opt, args.lr_decay)
            for opt in self.decoder_optimizers
        ]

        self.encoder_scheduler.last_epoch = self.start_epoch
        for scheduler in self.decoder_schedulers:
            scheduler.last_epoch = self.start_epoch

    def eval_batch(self, x, x_aug, dset_num):
        x, x_aug = x.float(), x_aug.float()

        z = self.encoder(x)
        y = self.decoders[dset_num](x, z)
        z_logits = self.discriminator(z)

        z_classification = torch.max(z_logits, dim=1)[1]

        z_accuracy = (z_classification == dset_num).float().mean()

        self.eval_d_right.add(z_accuracy.data.item())

        # discriminator_right = F.cross_entropy(z_logits, dset_num).mean()
        discriminator_right = F.cross_entropy(z_logits, torch.tensor(
            [dset_num] * x.size(0)).long().to(self.args.rank)).mean()
        recon_loss = cross_entropy_loss(y, x)

        self.evals_recon[dset_num].add(recon_loss.data.cpu().numpy().mean())

        total_loss = discriminator_right.data.item() * self.args.d_lambda + \
            recon_loss.mean().data.item()

        self.eval_total.add(total_loss)

        return total_loss

    def train_batch(self, x, x_aug, dset_num):
        x, x_aug = x.float(), x_aug.float()

        # Optimize D - discriminator right
        z = self.encoder(x)
        z_logits = self.discriminator(z)
        discriminator_right = F.cross_entropy(z_logits, torch.tensor(
            [dset_num] * x.size(0)).long().to(self.args.rank)).mean()
        loss = discriminator_right * self.args.d_lambda
        self.d_optimizer.zero_grad()
        loss.backward()
        if self.args.grad_clip is not None:
            clip_grad_value_(self.discriminator.parameters(),
                             self.args.grad_clip)

        self.d_optimizer.step()

        # optimize G - reconstructs well, discriminator wrong
        z = self.encoder(x_aug)
        y = self.decoders[dset_num](x, z)
        z_logits = self.discriminator(z)
        discriminator_wrong = - \
            F.cross_entropy(z_logits, torch.tensor(
                [dset_num] * x.size(0)).long().to(self.args.rank)).mean()

        if not (-100 < discriminator_right.data.item() < 100):
            self.logger.debug(f'z_logits: {z_logits.detach().cpu().numpy()}')
            self.logger.debug(f'dset_num: {dset_num}')

        recon_loss = cross_entropy_loss(y, x)
        self.losses_recon[dset_num].add(recon_loss.data.cpu().numpy().mean())

        loss = (recon_loss.mean() + self.args.d_lambda * discriminator_wrong)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizers[dset_num].zero_grad()
        loss.backward()
        if self.args.grad_clip is not None:
            clip_grad_value_(self.encoder.parameters(), self.args.grad_clip)
            clip_grad_value_(
                self.decoders[dset_num].parameters(), self.args.grad_clip)
        self.encoder_optimizer.step()
        self.decoder_optimizers[dset_num].step()

        self.loss_total.add(loss.data.item())

        return loss.data.item()

    def train_epoch(self, epoch):

        for meter in self.losses_recon:
            meter.reset()

        self.loss_d_right.reset()
        self.loss_total.reset()

        self.encoder.train()
        self.decoders.train()
        self.discriminator.train()

        n_batches = self.args.epoch_len

        with tqdm(total=n_batches, desc='Train epoch %d' % epoch) as train_enum:
            for _ in range(n_batches):
                x, x_aug, dset_num = next(self.data.train_iter)
                x = x.to(self.args.rank)
                x_aug = x_aug.to(self.args.rank)
                batch_loss = self.train_batch(x, x_aug, dset_num[0])

                train_enum.set_description(
                    f'Train (loss: {batch_loss:.2f}) epoch {epoch}')
                train_enum.update()

    def evaluate_epoch(self, epoch):
        for meter in self.evals_recon:
            meter.reset()
        self.eval_d_right.reset()
        self.eval_total.reset()

        self.encoder.eval()
        self.decoders.eval()
        self.discriminator.eval()

        n_batches = int(np.ceil(self.args.epoch_len / 10))

        with tqdm(total=n_batches) as valid_enum, torch.no_grad():
            for batch_num in range(n_batches):
                if self.args.short and batch_num == 10:
                    break

                x, x_aug, dset_num = next(self.data.valid_iter)
                x = x.to(self.args.rank)
                x_aug = x_aug.to(self.args.rank)
                batch_loss = self.eval_batch(x, x_aug, dset_num[0])

                valid_enum.set_description(
                    f'Test (loss: {batch_loss:.2f}) epoch {epoch}')
                valid_enum.update()

    @staticmethod
    def format_losses(meters):
        losses = [meter.summarize_epoch() for meter in meters]
        return ', '.join('{:.4f}'.format(x) for x in losses)

    def train_losses(self):
        meters = [*self.losses_recon, self.loss_d_right]
        return self.format_losses(meters)

    def eval_losses(self):
        meters = [*self.evals_recon, self.eval_d_right]
        return self.format_losses(meters)

    def train(self):
        best_eval = float('inf')

        # Begin!
        for epoch in range(self.start_epoch, self.start_epoch + self.args.epochs):
            self.logger.info(f'Starting epoch')
            self.train_epoch(epoch)
            self.evaluate_epoch(epoch)

            self.logger.info(f'Epoch %s Rank {self.args.rank} - Train loss: (%s), Test loss (%s)',
                             epoch, self.train_losses(), self.eval_losses())

            self.encoder_scheduler.step()
            for scheduler in self.decoder_schedulers:
                scheduler.step()
            val_loss = self.eval_total.summarize_epoch()

            if self.args.rank == 0:
                if val_loss < best_eval:
                    self.save_model(f'bestmodel.pth')
                    best_eval = val_loss

                if not self.args.per_epoch:
                    self.save_model(f'lastmodel.pth')
                else:
                    self.save_model(f'lastmodel_epoch_{epoch}.pth')

                torch.save([self.args,
                            epoch],
                           '%s/args.pth' % self.expPath)

            self.logger.debug('Ended epoch')

    def save_model(self, filename):
        save_path = self.expPath / filename

        torch.save({'encoder_state': self.encoder.module.state_dict(),
                    'decoder_state': [decoder.module.state_dict() for decoder in self.decoders],
                    'discriminator_state': self.discriminator.module.state_dict(),
                    'encoder_optimizer_state': self.encoder_optimizer.state_dict(),
                    'decoder_optimizer_state': [opt.state_dict() for opt in self.decoder_optimizers],
                    'd_optimizer_state': self.d_optimizer.state_dict()
                    },
                   save_path)

        self.logger.debug(f'Saved model to {save_path}')


def main():
    args = parser.parse_args()
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.rank = int(os.environ['RANK'])
    if args.rank == 0:
        args.is_master = True
    else:
        args.is_master = False

    print('Before init_process_group')
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url)

    Trainer(args).train()


if __name__ == '__main__':
    main()
