"""
Main training and evaluation script for CS-WAE
"""
import os
import torch
import lpips
from src.config import config
from src.models import SphericalWAE_Supervised
from src.datasets.mnist import get_mnist_loaders
from src.trainers.trainer import CSWAETrainer
from src.visualization.plots import plot_results, plot_slerp
from src.metrics.evaluation import ModelEvaluator


def main():
    """Main function to train and evaluate CS-WAE"""
    print("=" * 60)
    print("CS-WAE Training and Evaluation")
    print("=" * 60)
    
    # Create save directory
    save_dir = "results_cs_wae"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders()
    
    # Initialize model
    print(f"Initializing CS-WAE model on device: {config.device}")
    model = SphericalWAE_Supervised(
        latent_dim=config.latent_dim, 
        n_classes=config.n_classes
    ).to(config.device)
    
    # Initialize trainer
    trainer = CSWAETrainer(model, train_loader)
    
    # Train model
    print("Starting training...")
    history = trainer.train()
    
    # Save model
    model_path = f'{save_dir}/cs_wae_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_results(history, model, test_loader, save_dir=save_dir)
    plot_slerp(model, save_dir=save_dir)
    
    # Evaluate model
    print("Starting comprehensive evaluation...")
    evaluator = ModelEvaluator()
    metrics = evaluator.comprehensive_evaluation(
        model, 'CS-WAE', test_loader, save_dir
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for metric, value in metrics.items():
        print(f"{metric:<15}: {value:.4f}")
    print("=" * 60)
    
    print(f"\nAll results saved to: {save_dir}")
    print("Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
