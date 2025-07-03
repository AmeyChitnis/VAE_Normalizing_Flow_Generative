import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=20):
        super(VAE, self).__init__()
        # encoder
        self.fc1       = nn.Linear(input_dim,    hidden_dim)
        self.fc_mu     = nn.Linear(hidden_dim,   latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim,   latent_dim)
        # decoder
        self.fc3        = nn.Linear(latent_dim, hidden_dim)
        self.fc4_mu     = nn.Linear(hidden_dim, input_dim)
        self.fc4_logvar = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3       = F.relu(self.fc3(z))
        mu_dec   = self.fc4_mu(h3)
        logvar_d = self.fc4_logvar(h3)
        recon_x = torch.sigmoid(self.fc4_mu(h3))
        return recon_x  # Return a single tensor


    def forward(self, x):
        # x: [batch, input_dim]
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        recon_x    = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    MSE = F.binary_cross_entropy(recon_x, x.view(-1, 28*28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE, KLD

