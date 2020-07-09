from torch import reshape
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PrintShape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)

class ALAE(nn.Module):
    def __init__(self):
        super(ALAE, self).__init__()
        self.z_dim = 128
        self.latent_dim = 72
        self.gamma = 10

        self.lr = 0.0002
        self.beta1 = 0.0
        self.beta2 = 0.99

        self.to_latent = nn.Sequential(
            nn.Linear(self.z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.latent_dim),
        )

        self.generator = nn.Sequential(
            # (input_size - 1) * stride - 2 * padding + (kernal - 1) + 1
            # (2 - 1) * 2 - 2 * 0 + (3 - 1) + 1
            # 5
            View((8, 3, 3)),
            nn.ConvTranspose2d(8, 64, 4, stride=2),  # b, 64, 8, 8
            # PrintShape(),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 8, 4, stride=2, padding=1),  # b, 8, 16, 16
            # PrintShape(),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=3),  # b, 1, 28, 28
            # PrintShape(),
            nn.Sigmoid()
        )

        self.encoder = nn.Sequential(
            # [(input_size − kernal + 2 * padding) / stride] + 1
            # [(28 - 3 + 2 * 1) / 3] + 1
            # 10
            nn.Conv2d(1, 64, 4, stride=2, padding=1),  # b, 64, 14, 14
            # PrintShape(),
            nn.LeakyReLU(inplace = True),
            nn.MaxPool2d(2, stride=2),  # b, 64, 7, 7
            # PrintShape(),
            nn.Conv2d(64, 8, 4, stride=1, padding=1),  # b, 8, 6, 6
            # PrintShape(),
            nn.LeakyReLU(inplace = True),
            nn.MaxPool2d(2, stride=2),  # b, 8, 3, 3
            # PrintShape(),
            nn.Flatten(start_dim=1)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace = True),
            nn.Linear(1024, 1)
        )

        self.fakepass = nn.Sequential(self.to_latent, self.generator, self.encoder, self.discriminator)
        self.realpass = nn.Sequential(self.encoder, self.discriminator)
        self.latentpass = nn.Sequential(self.to_latent, self.generator, self.encoder)

        self.ed_opt = torch.optim.Adam(
            [
                {'params': self.encoder.parameters()},
                {'params': self.discriminator.parameters()}
            ],
            self.lr,
            betas=(self.beta1, self.beta2)
        )
        self.fg_opt = torch.optim.Adam(
            [
                {'params': self.to_latent.parameters()},
                {'params': self.generator.parameters()}
            ],
            self.lr,
            betas=(self.beta1, self.beta2)
        )
        self.eg_opt = torch.optim.Adam(
            [
                {'params': self.encoder.parameters()},
                {'params': self.generator.parameters()}
            ],
            self.lr,
            betas=(self.beta1, self.beta2)
        )
        self.to(device)

    def discriminator_loss(self, z, x):
        fake_loss = torch.mean(F.softplus(self.fakepass(z)), -1)
        real_loss = torch.mean(F.softplus(-self.realpass(x)), -1)
        real_grads = torch.autograd.grad(
            real_loss,
            x,
            grad_outputs=torch.ones_like(real_loss),
            create_graph=True,
            retain_graph=True
        )[0]
        r1_penalty = torch.sum(real_grads.pow(2.0), (1,2,3))
        loss = fake_loss + real_loss + r1_penalty * self.gamma / 2
        return loss

    def generator_loss(self, z):
        return torch.mean(F.softplus(-self.fakepass(z)), -1)

    def latent_loss(self, z):
        latent = torch.flatten(self.to_latent(z), start_dim=1)
        recovered = self.latentpass(z)
        return (latent - recovered).pow(2.0)

    def train_step(self, x, labels):
        bsize = x.shape[0]
        z = torch.normal(size=(bsize, self.z_dim - 10), mean=0, std=1, requires_grad=False).to(device)
        z = torch.cat([z, labels], dim=-1)
        z.requires_grad = True

        self.ed_opt.zero_grad()
        loss_d = self.discriminator_loss(z, x)
        loss_d.backward(torch.ones_like(loss_d))
        self.ed_opt.step()

        self.fg_opt.zero_grad()
        loss_g = self.generator_loss(z)
        loss_g.backward(torch.ones_like(loss_g))
        self.fg_opt.step()

        self.ed_opt.zero_grad()
        loss_l = self.latent_loss(z)
        loss_l.backward(torch.ones_like(loss_l))
        self.eg_opt.step()

        loss_d_float = loss_d.mean().detach().cpu().item()
        loss_g_float = loss_g.mean().detach().cpu().item()
        loss_l_float = loss_l.mean().detach().cpu().item()

        return {
            'loss_d': loss_d_float,
            'loss_g': loss_g_float,
            'loss_l': loss_l_float,
            'total_loss': loss_d_float + loss_g_float + loss_l_float
        }

    def fit(self, train_loader, epochs = 1):
        train_history = []
        for epoch in tqdm(range(epochs)):
            losses = {}
            for (img_tensors, target) in iter(train_loader):
                img_tensors = img_tensors.to(device)
                img_tensors.requires_grad = True
                labels = torch.tensor(np.eye(10)[target], requires_grad=False, dtype=torch.float).to(device)
                losses = self.train_step(img_tensors, labels)
            losses['epoch'] = epoch
            train_history.append(losses)
        return pd.DataFrame.from_records(train_history)