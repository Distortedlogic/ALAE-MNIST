import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda:0")

class ALAE(nn.Module):
    def __init__(self):
        super(ALAE, self).__init__()
        self.z_dim = 128
        self.latent_dim = 50
        self.output_dim = 784 + 10
        self.gamma = 10

        self.lr = 0.002
        self.beta1 = 0.0
        self.beta2 = 0.99

        self.f = nn.Sequential(
            nn.Linear(self.z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.latent_dim)
        )

        self.g = nn.Sequential(
            # nn.ConvTranspose2d(4, 16, 2, stride=2),
            # nn.ConvTranspose2d(16, 3, 2, stride=2),
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.output_dim)
        )
        self.e = nn.Sequential(
            # nn.Conv2d(1, 8, 3, padding=1),
            # nn.ReLU(True),
            # nn.MaxPool2d(2, 2),
            # nn.Conv2d(16, 4, 3, padding=1),
            # nn.ReLU(True),
            # nn.MaxPool2d(2, 2),
            nn.Linear(self.output_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.latent_dim)
        )

        self.d = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1)
        )

        self.fakepass = nn.Sequential(self.f, self.g, self.e, self.d)
        self.realpass = nn.Sequential(self.e, self.d)
        self.latentpass = nn.Sequential(self.f, self.g, self.e)

        self.ed_opt = torch.optim.Adam(
            [
                {'params': self.e.parameters()},
                {'params': self.d.parameters()}
            ],
            self.lr,
            betas=(self.beta1, self.beta2)
        )
        self.fg_opt = torch.optim.Adam(
            [
                {'params': self.f.parameters()},
                {'params': self.g.parameters()}
            ],
            self.lr,
            betas=(self.beta1, self.beta2)
        )
        self.eg_opt = torch.optim.Adam(
            [
                {'params': self.e.parameters()},
                {'params': self.g.parameters()}
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
        r1_penalty = torch.sum(real_grads.pow(2.0), -1)
        loss = fake_loss + real_loss + r1_penalty * self.gamma / 2
        return loss

    def generator_loss(self, z):
        return torch.mean(F.softplus(-self.fakepass(z)), -1)

    def latent_loss(self, z):
        latent = self.f(z)
        recovered = self.latentpass(z)
        return (latent - recovered).pow(2.0)

    def losses(self, x):
        bsize = x.shape[0]
        z = torch.normal(size=(bsize, self.z_dim), mean=0, std=1, requires_grad=True).to(device)

        loss_d = self.discriminator_loss(z, x),
        loss_g = self.generator_loss(z),
        loss_l = self.latent_loss(z)

        loss_d_float = loss_d.detach()[0].item()
        loss_g_float = loss_g.detach().item()
        loss_l_float = loss_l.detach().item()

        return {
            'loss_d': loss_d_float,
            'loss_g': loss_g_float,
            'loss_l': loss_l_float,
            'total_loss': loss_d_float + loss_g_float + loss_l_float
        }

    def train_step(self, x):
        bsize = x.shape[0]
        z = torch.normal(size=(bsize, self.z_dim), mean=0, std=1, requires_grad=True).to(device)

        self.ed_opt.zero_grad()
        loss_d = self.discriminator_loss(z, x)
        loss_d.backward(torch.ones_like(loss_d))
        self.ed_opt.step()

        self.fg_opt.zero_grad()
        loss_g = self.generator_loss(z)
        loss_g.backward(torch.ones_like(loss_g))
        self.fg_opt.step()

        self.eg_opt.zero_grad()
        loss_l = self.latent_loss(z)
        loss_l.backward(torch.ones_like(loss_l))
        self.eg_opt.step()

        loss_d_float = loss_d.detach()
        loss_g_float = loss_g.detach()
        loss_l_float = loss_l.detach()

        return {
            'loss_d': loss_d_float,
            'loss_g': loss_g_float,
            'loss_l': loss_l_float,
        }

    def fit(self, train_loader, test_loader, epochs = 1):
        train_history = []
        for epoch in tqdm(range(epochs)):
            losses = {}
            for idx, (img_tensors, target) in enumerate(train_loader):
                flat_img = torch.reshape(img_tensors.to(device), (-1, 784))
                labels = torch.tensor(np.eye(10)[target], dtype=torch.float).to(device)
                nn_input = torch.cat([flat_img, labels], dim=-1)
                nn_input.requires_grad = True
                losses = self.train_step(nn_input)

            losses['epoch'] = epoch
            train_history.append(losses)

            test_history = []
            # for idx, (img_tensors, target) in enumerate(test_loader):
            #     if idx > 10:
            #         break
            #     nn_input = torch.flatten(img_tensors, start_dim=1).to(device)
            #       nn_input.requires_grad = True
            #     losses = self.losses(nn_input)
            #     test_history.append(losses)
        return pd.DataFrame.from_records(train_history), pd.DataFrame.from_records(test_history)

