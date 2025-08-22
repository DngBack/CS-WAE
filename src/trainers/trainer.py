"""
Training functions for CS-WAE and baseline models
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips
from ..config import config
from ..utils.loss import calculate_cs_wae_loss


class CSWAETrainer:
    """Trainer class for CS-WAE model"""
    
    def __init__(self, model, train_loader, device=None):
        self.model = model
        self.train_loader = train_loader
        self.device = device or config.device
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)
        
        # Initialize LPIPS loss function
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(self.device)
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        loss_acc, recon_acc, sup_mmd_acc, unsup_mmd_acc = 0.0, 0.0, 0.0, 0.0

        # Calculate annealing weights
        anneal_rate = min(1.0, (epoch + 1) / config.anneal_epochs)
        current_sup_weight = config.sup_mmd_weight * anneal_rate
        current_unsup_weight = config.unsup_mmd_weight * anneal_rate

        for data, labels in pbar:
            data, labels = data.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            x_hat, z_q = self.model(data)

            # Calculate loss
            loss, recon, sup_mmd, unsup_mmd = calculate_cs_wae_loss(
                data, labels, x_hat, z_q, self.model, self.loss_fn_vgg, 
                current_sup_weight, current_unsup_weight
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            loss_acc += loss.item()
            recon_acc += recon.item()
            sup_mmd_acc += sup_mmd.item()
            unsup_mmd_acc += unsup_mmd.item()

            pbar.set_postfix({
                "Loss": f"{loss.item():.3f}", 
                "Recon": f"{recon.item():.3f}",
                "SupMMD": f"{sup_mmd.item():.4f}", 
                "UnsupMMD": f"{unsup_mmd.item():.4f}",
                "λ": f"{current_sup_weight:.2f}", 
                "γ": f"{current_unsup_weight:.2f}"
            })

        self.scheduler.step()
        n_batches = len(self.train_loader)
        avg_losses = (loss_acc/n_batches, recon_acc/n_batches, 
                     sup_mmd_acc/n_batches, unsup_mmd_acc/n_batches)
        
        print(f"====> Epoch {epoch+1} Avg Loss: Total={avg_losses[0]:.4f}, "
              f"Recon={avg_losses[1]:.4f}, SupMMD={avg_losses[2]:.4f}, "
              f"UnsupMMD={avg_losses[3]:.4f}")
        
        return avg_losses

    def train(self, epochs=None):
        """Full training loop"""
        epochs = epochs or config.epochs
        history = []
        
        print(f"Starting training on device: {self.device}")
        print(f"Configuration: {config.__dict__}")
        
        for epoch in range(epochs):
            avg_losses = self.train_epoch(epoch)
            history.append(avg_losses)
            
        print("Training completed!")
        return history


class BaselineTrainer:
    """Trainer class for baseline models"""
    
    def __init__(self, model, model_name, train_loader, device=None):
        self.model = model
        self.model_name = model_name
        self.train_loader = train_loader
        self.device = device or config.device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [{self.model_name}]")
        
        for data, _ in pbar:
            data = data.to(self.device)
            self.optimizer.zero_grad()

            if self.model_name == 'VAE':
                recon_batch, mu, log_var = self.model(data)
                loss = self.model.loss_function(recon_batch, data, mu, log_var)
            elif self.model_name == 'WAE-MMD':
                recon_batch, z = self.model(data)
                loss = self.model.loss_function(recon_batch, data, z)
            elif self.model_name == 'S-VAE':
                recon_batch, q_z, p_z = self.model(data)
                loss = self.model.loss_function(recon_batch, data, q_z, p_z)
            elif self.model_name == 'VaDE':
                recon_batch, mu, log_var, z = self.model(data)
                loss = self.model.loss_function(recon_batch, data, mu, log_var, z)

            loss.backward()
            self.optimizer.step()
            pbar.set_postfix({"Loss": loss.item() / len(data)})

    def train(self, epochs=None):
        """Full training loop"""
        epochs = epochs or config.epochs
        
        print(f"Starting training for {self.model_name} on device: {self.device}")
        
        for epoch in range(epochs):
            self.train_epoch(epoch)
            
        print(f"Training completed for {self.model_name}!")
