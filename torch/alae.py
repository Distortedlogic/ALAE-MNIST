import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

mse = nn.MSELoss()

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
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.output_dim)
        )

        self.e = nn.Sequential(
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

    def discriminator_loss(self, z, x):
        fake_loss = (nn.Softplus()(self.fakepass(z))).mean(dim=0)
        real_loss = (nn.Softplus()(-self.realpass(x))).mean(dim=0)
        real_grads = torch.autograd.grad(real_loss, x, create_graph=True, retain_graph=True)[0]
        r1_penalty = torch.sum(real_grads.pow(2.0), dim=[1, 2, 3])

        return fake_loss + real_loss + r1_penalty * self.gamma / 2

    def generator_loss(self, z):
        return F.softplus(-self.fakepass(z)).mean()

    def latent_loss(self, z):
        latent = self.f(z)
        recovered = self.latentpass(z)
        return mse(latent, recovered)

    def losses(x):
        bsize = x.shape[0]
        z = torch.normal(size=(bsize, self.z_dim))
        return {
            'loss_d': self.discriminator_loss(z, x),
            'loss_g': self.generator_loss(z),
            'loss_l': self.latent_loss(z)
        }

    def train_step(self, x):
        bsize = x.shape[0]
        z = torch.normal(size=(bsize, self.z_dim))

        self.ed_opt.zero_grad()
        loss_d = self.discriminator_loss(z, x)
        loss_d.backwards()
        self.ed_opt.step()

        self.fg_opt.zero_grad()
        loss_g = self.generator_loss(z)
        loss_g.backwards()
        self.fg_opt.step()

        self.ed_opt.zero_grad()
        loss_l = self.latent_loss(z)
        loss_l.backwards()
        self.eg_opt.step()

        return {
            'loss_d': loss_d,
            'loss_g': loss_g,
            'loss_l': loss_l
        }

    def train(self, epochs, train, test):
        step = 0
        for _ in tqdm(range(epochs)):
            for data in train:
                data = torch.as_tensor(data)
                data = torch.reshape(data, (data.shape[0], -1))
                self.train_step(data)
                step += 1

            metrics = []
            for data in test:
                data = torch.as_tensor(data)
                data = torch.reshape(data, (data.shape[0], -1))
                metrics.append(self.losses(data))

