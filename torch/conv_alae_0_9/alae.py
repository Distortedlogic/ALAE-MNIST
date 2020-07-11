from torch import reshape
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        # conv_output_size = [(input_size − kernal + 2 * padding) / stride] + 1
        # pool_output_size = (input size - kernal) / stride + 1
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=kwargs['stride']
        )
        self.batchnorm2d = nn.BatchNorm2d(out_channels)
        # self.maxpool2d = nn.MaxPool2d(kwargs['pool_size'], stride=kwargs['pool_stride'])

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batchnorm2d(x)
        x = F.leaky_relu(x, 0.2)
        # x = self.maxpool2d(x)
        return x

class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        # (input_size - 1) * stride - 2 * padding + (kernal - 1) + 1
        self.convtrans2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=kwargs['stride'],
            padding=kwargs['padding']
        )
        self.batchnorm2d = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.convtrans2d(x)
        x = self.batchnorm2d(x)
        x = F.leaky_relu(x, 0.2)
        return x

class z_map_nn(nn.Module):
    def __init__(self, z_dim, latent_dim):
        super(z_map_nn, self).__init__()
        self.lin1 = nn.Linear(z_dim, 1024)
        self.lin2 = nn.Linear(1024, 512)
        self.lin3 = nn.Linear(512, latent_dim)
    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        return x

class generator_nn(nn.Module):
    def __init__(self, latent_dim):
        super(generator_nn, self).__init__()
        self.fc = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.convTrans1 = ConvTranspose2d(256, 512, 4, stride=1, padding=1)
        self.convTrans2 = ConvTranspose2d(512, 256, 4, stride=2, padding=0)
        self.convTrans3 = ConvTranspose2d(256, 128, 4, stride=2, padding=0)
        self.convTrans4 = nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1)
    def forward(self, x):
        x = self.fc(x)
        x = self.fc2(x)
        x = x.view(*x.shape, 1, 1)
        x = self.convTrans1(x)
        x = self.convTrans2(x)
        x = self.convTrans3(x)
        x = self.convTrans4(x)
        return x

class encoder_nn(nn.Module):
    def __init__(self, latent_dim):
        super(encoder_nn, self).__init__()
        self.conv1 = Conv2d(1, 128, 4, stride=1)
        self.conv2 = Conv2d(128, 256, 4, stride=1)
        self.conv3 = Conv2d(256, 512, 4, stride=1)
        self.conv4 = Conv2d(512, 1, 4, stride=1)
        self.fc = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, latent_dim)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.fc2(x)
        return x

class discriminator_nn(nn.Module):
    def __init__(self, latent_dim):
        super(discriminator_nn, self).__init__()
        self.lin1 = nn.Linear(latent_dim, 1024)
        self.lin2 = nn.Linear(1024, 1024)
        self.lin3 = nn.Linear(1024, 1)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.batchnorm2 = nn.BatchNorm1d(1024)
    def forward(self, x):
        x = self.lin1(x)
        x = self.batchnorm1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.lin2(x)
        x = self.batchnorm2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.lin3(x)
        return x

class ALAE(nn.Module):
    def __init__(self):
        super(ALAE, self).__init__()
        self.z_dim = 100
        self.latent_dim = 50
        self.gamma = 10

        self.ed_lr = 0.0001
        self.fg_lr = 0.0004
        self.eg_lr = 0.0002
        self.beta1 = 0.0
        self.beta2 = 0.99

        self.z_map = z_map_nn(self.z_dim, self.latent_dim)
        self.generator = generator_nn(self.latent_dim)
        self.encoder = encoder_nn(self.latent_dim)
        self.discriminator = discriminator_nn(self.latent_dim)

        self.ed_opt = torch.optim.Adam(
            [
                {'params': self.encoder.parameters()},
                {'params': self.discriminator.parameters()}
            ],
            self.ed_lr,
            betas=(self.beta1, self.beta2)
        )
        self.fg_opt = torch.optim.Adam(
            [
                {'params': self.z_map.parameters()},
                {'params': self.generator.parameters()}
            ],
            self.fg_lr,
            betas=(self.beta1, self.beta2)
        )
        self.eg_opt = torch.optim.Adam(
            [
                {'params': self.generator.parameters()},
                {'params': self.encoder.parameters()}
            ],
            self.eg_lr,
            betas=(self.beta1, self.beta2)
        )
        self.fakepass = nn.Sequential(self.z_map, self.generator, self.encoder, self.discriminator)
        self.realpass = nn.Sequential(self.encoder, self.discriminator)
        self.latentpass = nn.Sequential(self.z_map, self.generator, self.encoder)
        self.to(device)

    def discriminator_loss(self, z, x):
        fake_loss = F.softplus(self.fakepass(z))
        real_loss = F.softplus(-self.realpass(x))
        real_grads = torch.autograd.grad(
            real_loss,
            x,
            grad_outputs=torch.ones_like(real_loss),
            create_graph=True,
            retain_graph=True
        )[0]
        penalty = torch.sum(real_grads.pow(2.0), dim=(2,3)) * self.gamma / 2
        loss = fake_loss + real_loss + penalty
        return loss

    def generator_loss(self, z):
        return F.softplus(-self.fakepass(z))

    def latent_loss(self, z):
        latent = self.z_map(z)
        recovered = self.latentpass(z)
        loss_reconst = torch.mean(torch.square(latent - recovered), dim=1)
        loss_kl_full = F.kl_div(F.log_softmax(latent, dim=1), F.softmax(recovered, dim=1), reduction='none')
        loss_kl = torch.sum(loss_kl_full, dim=1)
        loss = (loss_reconst + loss_kl)/2
        return loss, loss_reconst, loss_kl

    def sample_z(self, x):
        z = torch.randn(size=(x.shape[0], self.z_dim)).to(device)
        z.requires_grad = True
        return z

    def alae_train_step(self, x):
        self.ed_opt.zero_grad()
        loss_d = self.discriminator_loss(self.sample_z(x), x).mean()
        loss_d.backward(torch.ones_like(loss_d))
        self.ed_opt.step()

        self.fg_opt.zero_grad()
        loss_g = self.generator_loss(self.sample_z(x)).mean()
        loss_g.backward(torch.ones_like(loss_g))
        self.fg_opt.step()

        self.eg_opt.zero_grad()
        loss_l, loss_reconst, loss_kl = self.latent_loss(self.sample_z(x))
        loss_l.mean().backward(torch.ones_like(loss_l.mean()))
        self.eg_opt.step()

        loss_d_float = loss_d.mean().detach().cpu().item()
        loss_g_float = loss_g.mean().detach().cpu().item()
        loss_l_float = loss_l.mean().detach().cpu().item()

        return {
            'loss_d': loss_d_float,
            'loss_g': loss_g_float,
            'loss_l': loss_l_float,
            'loss_reconst': loss_reconst.mean().detach().cpu().item(),
            'loss_kl': loss_kl.mean().detach().cpu().item(),
            'total_loss': loss_d_float + loss_g_float + loss_l_float
        }

    def fit(self, train_loader, epochs = 1):
        train_history = []
        train_history = []
        epoch_history = []
        for epoch in tqdm(range(epochs)):
            losses = {}
            for (img_tensors, target) in iter(train_loader):
                img_tensors = img_tensors.to(device)
                img_tensors.requires_grad = True
                losses = self.alae_train_step(img_tensors)
                losses['epoch'] = epoch
                train_history.append(losses)
            epoch_history.append(losses)
        return pd.DataFrame.from_records(train_history), pd.DataFrame.from_records(epoch_history)