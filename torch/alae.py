import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

    def discriminator_loss(self, z, x):
        fake_loss = (nn.Softplus()(self.fakepass(z))).mean(dim=0)
        real_loss = (nn.Softplus()(-self.realpass(x))).mean(dim=0)
        real_grads = torch.autograd.grad(real_loss, x, create_graph=True, retain_graph=True)[0]
        r1_penalty = torch.sum(real_grads.pow(2.0), dim=[1, 2, 3])
        
        return fake_loss + real_loss + r1_penalty * (self.gamma * 0.5)

    def generator_loss(self, z):
        return F.softplus(-self.fakepass(z)).mean()

    def latent_loss(self, z):
        latent = self.f(z)
        recovered = self.latentpass(z)
        return ((latent - recovered)**2).mean(dim=0)

    def losses(self, x):
        bsize = x.shape[0]
        z = tf.random.normal((bsize, self.z_dim), 0, 1)
        return {
            'disc': self._disc_loss(z, x).numpy(),
            'gen': self._gen_loss(z).numpy(),
            'latent': self._latent_loss(z).numpy(),
        }