"""
Baseline comparison script for evaluating CS-WAE against other methods
"""
import os
import torch
import warnings
warnings.filterwarnings('ignore')

from src.config import config
from src.models import VAE, WAE_MMD, S_VAE, VaDE, SphericalWAE_Supervised
from src.datasets.mnist import get_mnist_loaders
from src.trainers.trainer import BaselineTrainer, CSWAETrainer
from src.metrics.evaluation import ModelEvaluator, create_comparison_table


def main():
    """Main function to compare CS-WAE with baseline methods"""
    print("=" * 80)
    print("CS-WAE vs Baselines Comparison")
    print("=" * 80)
    
    # Create results directory
    results_dir = "baseline_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders()
    
    # Define models to compare
    models_to_run = {
        "VAE": VAE(config.latent_dim),
        "WAE-MMD": WAE_MMD(config.latent_dim),
        # "S-VAE": S_VAE(config.latent_dim),  # Uncomment if hyperspherical_vae is available
        # "VaDE": VaDE(config.latent_dim, config.n_classes),  # Uncomment if needed
        "CS-WAE": SphericalWAE_Supervised(config.latent_dim, config.n_classes)
    }
    
    # Store all results
    all_results = {}
    evaluator = ModelEvaluator()
    
    # Train and evaluate each model
    for model_name, model in models_to_run.items():
        print(f"\n{'='*30}")
        print(f"Training {model_name}")
        print(f"{'='*30}")
        
        model.to(config.device)
        
        # Train model
        if model_name == "CS-WAE":
            trainer = CSWAETrainer(model, train_loader)
            trainer.train(epochs=10)  # Reduced epochs for faster comparison
        else:
            trainer = BaselineTrainer(model, model_name, train_loader)
            trainer.train(epochs=10)  # Reduced epochs for faster comparison
        
        # Save model
        model_path = os.path.join(results_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
        
        # Evaluate model
        print(f"Evaluating {model_name}...")
        metrics = evaluator.comprehensive_evaluation(
            model, model_name, test_loader, results_dir
        )
        all_results[model_name] = metrics
        
        print(f"Completed {model_name}")
    
    # Create comparison table
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    comparison_df = create_comparison_table(all_results)
    
    # Save results to CSV
    comparison_df.to_csv(os.path.join(results_dir, "comparison_results.csv"))
    print(f"\nResults saved to: {results_dir}/comparison_results.csv")
    
    print("Comparison completed successfully!")


if __name__ == "__main__":
    main()
