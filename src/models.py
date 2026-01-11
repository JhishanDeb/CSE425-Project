import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== BASIC VAE ====================
class VAE(nn.Module):
    def __init__(self, input_dim=144, hidden_dims=[512, 256], latent_dim=32, dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# ==================== BETA-VAE ====================
class BetaVAE(VAE):
    def __init__(self, beta=4.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

# ==================== CONDITIONAL VAE (CVAE) ====================
class CVAE(nn.Module):
    def __init__(self, input_dim=144, condition_dim=10, latent_dim=32, hidden_dims=[512, 256]):
        super().__init__()
        enc_in = input_dim + condition_dim
        dec_in = latent_dim + condition_dim

        # Encoder
        layers = []
        prev = enc_in
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.mu = nn.Linear(prev, latent_dim)
        self.logvar = nn.Linear(prev, latent_dim)

        # Decoder
        layers = []
        prev = dec_in
        for h in reversed(hidden_dims):
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)]
            prev = h
        layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x, c):
        xc = torch.cat([x, c], dim=1)
        h = self.encoder(xc)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(logvar.exp()) * logvar.exp().sqrt()
        zc = torch.cat([z, c], dim=1)
        recon = self.decoder(zc)
        return recon, mu, logvar

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Scale KL by beta, typically beta > 1 for disentanglement
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

# ==================== CONVOLUTIONAL VAE (1D) ====================
class ConvVAE(nn.Module):
    """
    1D Convolutional VAE for temporal data (segments).
    Input: (Batch, Channels=24, Length=100) or similar.
    Features: Timbre(12) + Chroma(12) = 24 channels.
    """
    def __init__(self, input_channels=24, input_length=100, latent_dim=32, filters=[32, 64, 128], kernel_sizes=[5, 5, 3]):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        enc_layers = []
        in_ch = input_channels
        for f, k in zip(filters, kernel_sizes):
            enc_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, f, kernel_size=k, stride=2, padding=1),
                    nn.BatchNorm1d(f),
                    nn.ReLU(),
                )
            )
            in_ch = f
        self.encoder = nn.Sequential(*enc_layers)
        
        # Calculate flattened size
        # L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
        # Assuming Length=100:
        # L1: (100+2 - 5)/2 + 1 = 49
        # L2: (49+2 - 5)/2 + 1 = 23
        # L3: (23+2 - 3)/2 + 1 = 11
        # Final flat: 128 * 11 = 1408
        
        # Dynamically calculate size
        with torch.no_grad():
             dummy = torch.zeros(1, input_channels, input_length)
             dummy_out = self.encoder(dummy)
             self.flat_dim = dummy_out.view(1, -1).size(1)

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        
        # Decoder (Upsample + Conv to ensure exact output size)
        rev_filters = filters[::-1] # [128, 64, 32]
        # Target sizes for Upsampling to reverse [100 -> 49 -> 23 -> 11]
        # We need [11 -> 23 -> 49 -> 100]
        # Ideally we'd store these during forward or compute dynamically, 
        # but for this specific architecture (frames=100) we can hardcode or be approximate.
        # Better robust way: Use Scale Factor or simple fixed upsampling if we know the input size.
        
        self.dec_input_ch = rev_filters[0]
        self.decoder_layers = nn.ModuleList()
        
        # 128 -> 64 (Target 23)
        self.decoder_layers.append(
            nn.Sequential(
                nn.Upsample(size=23, mode='nearest'),
                nn.Conv1d(rev_filters[0], rev_filters[1], kernel_size=3, padding=1), 
                nn.BatchNorm1d(rev_filters[1]), nn.ReLU()
            )
        )
        # 64 -> 32 (Target 49)
        self.decoder_layers.append(
             nn.Sequential(
                nn.Upsample(size=49, mode='nearest'),
                nn.Conv1d(rev_filters[1], rev_filters[2], kernel_size=5, padding=2),
                nn.BatchNorm1d(rev_filters[2]), nn.ReLU()
            )
        )
        # 32 -> input_channels (Target 100)
        self.decoder_layers.append(
             nn.Sequential(
                nn.Upsample(size=input_length, mode='nearest'),
                nn.Conv1d(rev_filters[2], input_channels, kernel_size=5, padding=2),
                # No BatchNorm/ReLU at final layer typically
            )
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(h.size(0), self.dec_input_ch, -1) # (B, 128, ~11)
        # Note: The linear layer maps to self.flat_dim which matches encoder output (11)
        # So we start with length 11.
        
        for layer in self.decoder_layers:
            h = layer(h)
            
        return h

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# ==================== HYBRID VAE (Audio + Text) ====================
class HybridVAE(nn.Module):
    """
    Fuses Audio (Conv1D) and Text (MLP) components.
    """
    def __init__(self, audio_conf={}, text_conf={}, latent_dim=32, fusion_dim=64):
        super().__init__()
        
        # --- Audio Branch (Conv) ---
        # Instead of full ConvVAE, we just take the encoder part usually
        # But for simplicity, let's instantiate the ConvVAE encoder logic here or reuse the class
        # Let's reuse the ConvVAE class but we only use its encoder
        self.audio_net = ConvVAE(**audio_conf)
        self.audio_dim = self.audio_net.flat_dim # e.g. 1408
        
        # --- Text Branch (MLP) ---
        text_input_dim = text_conf.get('input_dim', 5000)
        text_hidden = text_conf.get('hidden_dim', 512)
        
        self.text_encoder = nn.Sequential(
            nn.Linear(text_input_dim, text_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(text_hidden),
            nn.Linear(text_hidden, 256),
            nn.ReLU()
        )
        self.text_dim = 256
        
        # --- Fusion ---
        self.fusion_fc = nn.Sequential(
            nn.Linear(self.audio_dim + self.text_dim, fusion_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim)
        )
        
        self.fc_mu = nn.Linear(fusion_dim, latent_dim)
        self.fc_logvar = nn.Linear(fusion_dim, latent_dim)
        
        # --- Decoders ---
        # We decode back to BOTH modalities
        
        # Shared decode latent mapping
        self.dec_fc = nn.Linear(latent_dim, fusion_dim)
        
        # Audio Decode branch
        self.audio_dec_fc = nn.Linear(fusion_dim, self.audio_dim)
        # We reuse the conv decoder layers from self.audio_net
        # self.audio_net.decoder_convs
        
        # Text Decode branch
        self.text_decoder = nn.Sequential(
            nn.Linear(fusion_dim, text_hidden),
            nn.ReLU(),
            nn.Linear(text_hidden, text_input_dim)
            # No sigmoid if using logits, or sigmoid if binary BoW
        )

    def encode(self, x_audio, x_text):
        # Audio
        h_a = self.audio_net.encoder(x_audio)
        h_a = h_a.view(h_a.size(0), -1)
        
        # Text
        h_t = self.text_encoder(x_text)
        
        # Concat
        h_cat = torch.cat([h_a, h_t], dim=1)
        
        # Fuse
        h_f = self.fusion_fc(h_cat)
        
        return self.fc_mu(h_f), self.fc_logvar(h_f)

    def decode(self, z):
        h_f = self.dec_fc(z)
        
        # Audio Recon
        h_a_rec = self.audio_dec_fc(h_f)
        h_a_rec = h_a_rec.view(h_a_rec.size(0), self.audio_net.dec_input_ch, -1)
        recon_audio = h_a_rec
        for layer in self.audio_net.decoder_layers:
            recon_audio = layer(recon_audio)
            
        # Text Recon
        recon_text = self.text_decoder(h_f)
        
        return recon_audio, recon_text

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_audio, x_text):
        mu, logvar = self.encode(x_audio, x_text)
        z = self.reparameterize(mu, logvar)
        recon_audio, recon_text = self.decode(z)
        return recon_audio, recon_text, mu, logvar

# ==================== AUTOENCODER (Baseline) ====================
class Autoencoder(nn.Module):
    """
    Standard Autoencoder (no KL divergence, deterministic bottleneck)
    """
    def __init__(self, input_dim=144, hidden_dims=[512, 256], latent_dim=32):
        super().__init__()
        # Encoder
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.fc_z = nn.Linear(prev, latent_dim)

        # Decoder
        layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)]
            prev = h
        layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_z(h) # direct Z, no mu/logvar

    def forward(self, x):
        z = self.encode(x)
        recon = self.decoder(z)
        return recon, z

def autoencoder_loss(recon, x):
    return F.mse_loss(recon, x, reduction='sum')

def hybrid_vae_loss(recon_audio, x_audio, recon_text, x_text, mu, logvar, beta=1.0, alpha=1.0):
    """
    alpha: weight for text loss vs audio loss
    """
    # Audio Loss (MSE)
    # Ensure shapes match for MSE - sometimes broadcasting happens if dimension 1 is squeezed
    # x_audio: (B, 24, 100), recon_audio: (B, 24, 100)
    if recon_audio.shape != x_audio.shape:
        # Try to fix scalar mismatches if any
        pass 
        
    recon_loss_audio = F.mse_loss(recon_audio, x_audio, reduction='sum')
    
    # Text Loss (MSE for TF-IDF features usually fine, or BCE if binary)
    # Using MSE for TF-IDF
    recon_loss_text = F.mse_loss(recon_text, x_text, reduction='sum')
    
    # KL Loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_recon = recon_loss_audio + alpha * recon_loss_text
    return total_recon + beta * kl_loss, recon_loss_audio, recon_loss_text, kl_loss

