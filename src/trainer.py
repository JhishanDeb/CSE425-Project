import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
import numpy as np
from src.models import vae_loss

class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"   Early stopping triggered!")
                self.early_stop = True

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, path)
    print(f"   → Checkpoint saved: {path.name}")

class VAETrainer:
    def __init__(self, model, train_loader, val_loader, config, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # Hardcoded defaults or from config if available
        # Ideally use config values if provided, but let's stick to safe defaults or passed config
        
        lr_scheduler_params = getattr(config, 'LR_SCHEDULER', {'factor': 0.5, 'patience': 7, 'min_lr': 1e-6})
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=lr_scheduler_params['factor'],
            patience=lr_scheduler_params['patience'], min_lr=lr_scheduler_params['min_lr']
        )
        
        patience = getattr(config, 'EARLY_STOPPING_PATIENCE', 15)
        self.early_stopping = EarlyStopping(patience=patience, verbose=True)

    def train_epoch(self):
        self.model.train()
        total_loss = total_recon = total_kl = 0
        for batch_x in tqdm(self.train_loader, desc="Training", leave=False):
            batch_x = batch_x[0].to(self.device) if isinstance(batch_x, (list, tuple)) else batch_x.to(self.device)

            self.optimizer.zero_grad()
            recon, mu, logvar = self.model(batch_x)
            
            # Identify beta if BetaVAE
            beta = getattr(self.model, 'beta', 1.0)
            loss, recon_l, kl_l = vae_loss(recon, batch_x, mu, logvar, beta=beta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_recon += recon_l.item()
            total_kl += kl_l.item()

        n = len(self.train_loader.dataset)
        return total_loss/n, total_recon/n, total_kl/n

    def validate(self):
        self.model.eval()
        total_loss = total_recon = total_kl = 0
        with torch.no_grad():
            for batch_x in self.val_loader:
                batch_x = batch_x[0].to(self.device) if isinstance(batch_x, (list, tuple)) else batch_x.to(self.device)
                recon, mu, logvar = self.model(batch_x)
                
                beta = getattr(self.model, 'beta', 1.0)
                loss, recon_l, kl_l = vae_loss(recon, batch_x, mu, logvar, beta=beta)
                
                total_loss += loss.item()
                total_recon += recon_l.item()
                total_kl += kl_l.item()

        n = len(self.val_loader.dataset)
        return total_loss/n, total_recon/n, total_kl/n

    def train(self, epochs=100, save_path=None):
        print(f"\\nTRAINING VAE FOR {epochs} EPOCHS")
        print(f"Device: {str(self.device).upper()} | Batch size: {self.train_loader.batch_size}")
        print(f"Train songs: {len(self.train_loader.dataset)} | Val songs: {len(self.val_loader.dataset)}")

        best_val = float('inf')
        start_time = time.time()

        for epoch in range(epochs):
            train_loss, train_r, train_kl = self.train_epoch()
            val_loss, val_r, val_kl = self.validate()

            self.scheduler.step(val_loss)
            lr = self.optimizer.param_groups[0]['lr']

            print(f"\\nEpoch {epoch+1}/{epochs} | {time.time()-start_time:.0f}s")
            print(f"   Train → Loss: {train_loss:.3f} (R: {train_r:.1f}, KL: {train_kl:.1f})")
            print(f"   Val   → Loss: {val_loss:.3f} (R: {val_r:.1f}, KL: {val_kl:.1f}) | LR: {lr:.2e}")

            if val_loss < best_val:
                best_val = val_loss
                if save_path:
                    save_checkpoint(self.model, self.optimizer, epoch, val_loss, save_path)

            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                break

        print(f"\\nTraining finished! Best Val Loss: {best_val:.3f}")

class HybridVAETrainer(VAETrainer):
    def __init__(self, model, train_loader, val_loader, config, device='cpu'):
        super().__init__(model, train_loader, val_loader, config, device)
        from src.models import hybrid_vae_loss # Late import to avoid circular dep if any
        self.loss_fn = hybrid_vae_loss

    def train_epoch(self):
        self.model.train()
        total_loss = total_audio = total_text = total_kl = 0
        
        for batch in tqdm(self.train_loader, desc="Hybrid Training", leave=False):
            # Batch is [audio, text]
            x_audio = batch[0].to(self.device)
            x_text  = batch[1].to(self.device)

            self.optimizer.zero_grad()
            
            # Forward
            recon_audio, recon_text, mu, logvar = self.model(x_audio, x_text)
            
            beta = getattr(self.model, 'beta', 1.0)
            
            loss, l_audio, l_text, l_kl = self.loss_fn(
                recon_audio, x_audio, 
                recon_text, x_text, 
                mu, logvar, 
                beta=beta
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_audio += l_audio.item()
            total_text += l_text.item()
            total_kl += l_kl.item()

        n = len(self.train_loader.dataset)
        return total_loss/n, total_audio/n, total_text/n, total_kl/n

    def validate(self):
        self.model.eval()
        total_loss = total_audio = total_text = total_kl = 0
        with torch.no_grad():
            for batch in self.val_loader:
                x_audio = batch[0].to(self.device)
                x_text  = batch[1].to(self.device)
                
                recon_audio, recon_text, mu, logvar = self.model(x_audio, x_text)
                
                beta = getattr(self.model, 'beta', 1.0)
                loss, l_audio, l_text, l_kl = self.loss_fn(
                    recon_audio, x_audio, 
                    recon_text, x_text, 
                    mu, logvar, 
                    beta=beta
                )
                
                total_loss += loss.item()
                total_audio += l_audio.item()
                total_text += l_text.item()
                total_kl += l_kl.item()

        n = len(self.val_loader.dataset)
        return total_loss/n, total_audio/n, total_text/n, total_kl/n

    def train(self, epochs=100, save_path=None):
        print(f"\\nTRAINING HYBRID VAE FOR {epochs} EPOCHS")
        print(f"Device: {str(self.device).upper()} | Batch size: {self.train_loader.batch_size}")
        
        best_val = float('inf')
        start_time = time.time()

        for epoch in range(epochs):
            t_loss, t_aud, t_txt, t_kl = self.train_epoch()
            v_loss, v_aud, v_txt, v_kl = self.validate()

            self.scheduler.step(v_loss)
            lr = self.optimizer.param_groups[0]['lr']

            print(f"\\nEpoch {epoch+1}/{epochs} | {time.time()-start_time:.0f}s")
            print(f"   Train → Loss: {t_loss:.1f} (Aud: {t_aud:.1f}, Txt: {t_txt:.1f}, KL: {t_kl:.1f})")
            print(f"   Val   → Loss: {v_loss:.1f} (Aud: {v_aud:.1f}, Txt: {v_txt:.1f}, KL: {v_kl:.1f}) | LR: {lr:.2e}")

            if v_loss < best_val:
                best_val = v_loss
                if save_path:
                    save_checkpoint(self.model, self.optimizer, epoch, v_loss, save_path)

            self.early_stopping(v_loss)
            if self.early_stopping.early_stop:
                break
