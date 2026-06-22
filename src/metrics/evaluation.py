"""
Comprehensive evaluation metrics for CS-WAE and baseline models
Includes FID, SSIM, PSNR, LPIPS, clustering accuracy, NMI, and ARI
"""
import os
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, accuracy_score, adjusted_rand_score
import pandas as pd

# External evaluation libraries
try:
    from pytorch_fid import fid_score
    import lpips as lpips_lib
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    EVAL_LIBS_AVAILABLE = True
except ImportError:
    EVAL_LIBS_AVAILABLE = False
    print("Warning: Some evaluation libraries are missing. Please install: pytorch-fid, lpips, torchmetrics")

from ..config import config
try:
    from ..config_ablation import ablation_config
except ImportError:
    ablation_config = None
from ..utils.utils import sample_uniform_sphere, mobius_reparam


class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, device=None, dataset="mnist", data_dir="./data"):
        self.device = device or config.device
        self.dataset = dataset
        self.data_dir = data_dir
        
        if EVAL_LIBS_AVAILABLE:
            self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
            self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
            self.lpips_metric = lpips_lib.LPIPS(net='vgg').to(self.device)
        
    def evaluate_reconstruction_quality(self, model, test_loader):
        """Evaluate reconstruction quality using SSIM, PSNR, and LPIPS"""
        if not EVAL_LIBS_AVAILABLE:
            return {"SSIM": 0.0, "PSNR": 0.0, "LPIPS": 0.0}
            
        model.eval()
        total_lpips_score = 0.0
        
        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc="Evaluating reconstruction"):
                images = images.to(self.device)
                
                # Get reconstructions
                if hasattr(model, 'encode_to_distribution'):
                    # CS-WAE model (both ablation and regular)
                    model_output = model(images)
                    if len(model_output) == 4:
                        # Ablation model returns (x_hat, z_q, mu_q, param_q)
                        reconstructed_images = model_output[0]
                    else:
                        # Regular CS-WAE returns (x_hat, other_stuff)
                        reconstructed_images = model_output[0]
                else:
                    # Baseline models
                    if hasattr(model, 'reparameterize'):  # VAE or VaDE
                        reconstructed_images = model(images)[0]
                    else:  # WAE-MMD or S-VAE
                        reconstructed_images = model(images)[0]
                
                # Update metrics
                self.ssim_metric.update(reconstructed_images, images)
                self.psnr_metric.update(reconstructed_images, images)
                
                # Calculate LPIPS
                images_lpips = (images * 2 - 1).repeat(1, 3, 1, 1)
                recon_lpips = (reconstructed_images * 2 - 1).repeat(1, 3, 1, 1)
                total_lpips_score += self.lpips_metric(images_lpips, recon_lpips).sum().item()
        
        final_ssim = self.ssim_metric.compute().item()
        final_psnr = self.psnr_metric.compute().item()
        final_lpips = total_lpips_score / len(test_loader.dataset)
        
        return {"SSIM": final_ssim, "PSNR": final_psnr, "LPIPS": final_lpips}
    
    def evaluate_clustering_quality(self, model, test_loader, model_name):
        """Evaluate clustering quality using ACC, NMI, and ARI"""
        model.eval()
        all_latents = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Extracting latent representations"):
                images = images.to(self.device)
                
                # Extract latent representations
                if model_name == 'CS-WAE' or hasattr(model, 'encode_to_distribution'):
                    # CS-WAE models (both regular and ablation)
                    mu_q, _ = model.encode_to_distribution(images)
                    latent_vectors = mu_q
                elif model_name in ['VAE', 'VaDE']:
                    mu, _ = model.encoder(images)
                    latent_vectors = mu
                elif model_name == 'WAE-MMD':
                    latent_vectors = model.encoder(images)
                elif model_name == 'S-VAE':
                    q_params = model.encoder(images)
                    latent_vectors = torch.nn.functional.normalize(q_params[:, :-1], p=2, dim=1)
                
                all_latents.append(latent_vectors.cpu().numpy())
                all_labels.append(labels.numpy())
        
        latents_np = np.concatenate(all_latents, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)
        
        # Perform K-means clustering
        current_config = ablation_config if ablation_config else config
        n_clusters = current_config.n_classes
        
        if model_name == 'VaDE' and hasattr(model, 'mu_c'):
            # Use VaDE cluster centers as initialization
            cluster_centers = model.mu_c.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters, init=cluster_centers, n_init=1)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        
        cluster_preds = kmeans.fit_predict(latents_np)
        
        # Calculate metrics
        nmi_score = normalized_mutual_info_score(labels_np, cluster_preds)
        ari_score = adjusted_rand_score(labels_np, cluster_preds)
        
        # Calculate clustering accuracy using Hungarian algorithm
        contingency_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
        for i in range(len(labels_np)):
            contingency_matrix[cluster_preds[i], labels_np[i]] += 1
        
        row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
        acc_score = contingency_matrix[row_ind, col_ind].sum() / len(labels_np)
        
        return {"ACC": acc_score, "NMI": nmi_score, "ARI": ari_score}
    
    def evaluate_generation_quality(self, model, model_name, save_dir):
        """Evaluate generation quality using FID"""
        if not EVAL_LIBS_AVAILABLE:
            return {"FID": 0.0}
            
        print(f"Calculating FID for {model_name} (this may take several minutes)...")
        
        # Create directories for real and generated images
        real_img_dir = os.path.join(save_dir, f"fid_images_{model_name}", "real")
        gen_img_dir = os.path.join(save_dir, f"fid_images_{model_name}", "generated")
        os.makedirs(real_img_dir, exist_ok=True)
        os.makedirs(gen_img_dir, exist_ok=True)
        
        # Save real images (only if not already done)
        if not os.listdir(real_img_dir):
            from ..datasets.loaders import build_test_dataset

            test_dataset = build_test_dataset(self.dataset, self.data_dir)
            
            for i, (img, _) in enumerate(tqdm(test_dataset, desc="Saving real images")):
                current_config = ablation_config if ablation_config else config
                if i >= current_config.num_images_for_fid:
                    break
                torchvision.utils.save_image(img.repeat(3, 1, 1), 
                                           os.path.join(real_img_dir, f"real_{i}.png"))
        
        # Generate and save fake images
        model.eval()
        generated_count = 0
        
        with torch.no_grad():
            current_config = ablation_config if ablation_config else config
            while generated_count < current_config.num_images_for_fid:
                num_to_gen = min(current_config.batch_size, current_config.num_images_for_fid - generated_count)
                
                # Generate samples based on model type
                if model_name == 'CS-WAE' or hasattr(model, 'encode_to_distribution'):
                    # Sample from CS-WAE priors (both regular and ablation)
                    # Use appropriate config based on model type
                    current_config = ablation_config if hasattr(model, 'variant_config') else config
                    random_classes = torch.randint(0, current_config.n_classes, (num_to_gen,), device=self.device)
                    
                    if hasattr(model, 'sample_from_prior'):
                        # Ablation model with sample_from_prior method
                        z_p_list = []
                        for i, class_idx in enumerate(random_classes):
                            z_sample = model.sample_from_prior(class_idx.item(), 1, self.device)
                            z_p_list.append(z_sample)
                        z_p = torch.cat(z_p_list, dim=0)
                    else:
                        # Regular CS-WAE model
                        normalized_prior_mus = torch.nn.functional.normalize(model.prior_mus, p=2, dim=1)
                        eps = sample_uniform_sphere(num_to_gen, current_config.latent_dim, device=self.device)
                        z_p = mobius_reparam(eps, normalized_prior_mus[random_classes], 
                                           torch.full((num_to_gen,), model.rho_p, device=self.device))
                    generated_images = model.decoder(z_p)
                elif model_name in ['VAE', 'WAE-MMD']:
                    # Sample from standard Gaussian
                    z = torch.randn(num_to_gen, config.latent_dim, device=self.device)
                    generated_images = model.decoder(z)
                elif model_name == 'S-VAE':
                    try:
                        from hyperspherical_vae.distributions import HypersphericalUniform
                        z = HypersphericalUniform(config.latent_dim, device=self.device).sample((num_to_gen,))
                        generated_images = model.decoder(z)
                    except ImportError:
                        print("Warning: Cannot generate S-VAE samples, hyperspherical_vae not available")
                        return {"FID": float('inf')}
                elif model_name == 'VaDE':
                    # Sample from VaDE GMM prior
                    pi_dist = torch.distributions.Categorical(torch.nn.functional.softmax(model.pi, dim=0))
                    c = pi_dist.sample((num_to_gen,))
                    mu_c = model.mu_c[c]
                    log_var_c = model.log_var_c[c]
                    z = model.reparameterize(mu_c, log_var_c)
                    generated_images = model.decoder(z)
                
                # Save generated images
                for i in range(generated_images.size(0)):
                    if generated_count >= current_config.num_images_for_fid:
                        break
                    image_rgb = generated_images[i].cpu().repeat(3, 1, 1)
                    torchvision.utils.save_image(image_rgb, 
                                               os.path.join(gen_img_dir, f"gen_{generated_count}.png"))
                    generated_count += 1
        
        # Calculate FID
        fid_value = fid_score.calculate_fid_given_paths(
            [real_img_dir, gen_img_dir], 
            batch_size=50, 
            device=self.device, 
            dims=2048
        )
        
        return {"FID": fid_value}
    
    def comprehensive_evaluation(self, model, model_name, test_loader, save_dir):
        """Run comprehensive evaluation on a model"""
        print(f"\n--- Starting comprehensive evaluation for {model_name} ---")
        
        # Evaluate reconstruction quality
        recon_metrics = self.evaluate_reconstruction_quality(model, test_loader)
        
        # Evaluate clustering quality
        clustering_metrics = self.evaluate_clustering_quality(model, test_loader, model_name)
        
        # Evaluate generation quality
        generation_metrics = self.evaluate_generation_quality(model, model_name, save_dir)
        
        # Combine all metrics
        all_metrics = {**recon_metrics, **clustering_metrics, **generation_metrics}
        
        print(f"Evaluation completed for {model_name}")
        return all_metrics


def create_comparison_table(all_results):
    """Create a comparison table from evaluation results"""
    df = pd.DataFrame(all_results).T
    df = df[["ACC", "NMI", "ARI", "FID", "LPIPS", "SSIM", "PSNR"]]
    
    print("\n\n" + " COMPREHENSIVE PERFORMANCE COMPARISON ".center(80, "="))
    print(df.to_markdown(floatfmt=".4f"))
    print("=" * 80)
    
    return df
