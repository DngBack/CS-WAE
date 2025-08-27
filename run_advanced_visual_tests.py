"""
Example script demonstrating how to use the advanced visual tests for CS-WAE
"""
import os
import torch
from src.config import config
from src.models import SphericalWAE_Supervised
from src.datasets.mnist import get_mnist_loaders
from src.visualization.advanced_tests import run_all_advanced_tests

def run_advanced_visual_analysis(model_path, save_dir="advanced_analysis_results"):
    """
    Run comprehensive visual analysis on a trained CS-WAE model.
    
    Args:
        model_path (str): Path to the trained model file (.pth)
        save_dir (str): Directory to save analysis results
    """
    print("🎨 Advanced Visual Analysis for CS-WAE")
    print("=" * 50)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders()
    
    # Get test dataset for single sample access
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = SphericalWAE_Supervised(
        latent_dim=config.latent_dim,
        n_classes=config.n_classes
    ).to(config.device)
    
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()
    
    print(f"Model loaded successfully on device: {config.device}")
    
    # Run all advanced tests
    results = run_all_advanced_tests(
        model=model,
        test_dataset=test_dataset,
        test_loader=test_loader,
        save_dir=save_dir,
        device=config.device
    )
    
    print("\n📊 Analysis Summary:")
    print(f"Total files generated: {len(results)}")
    for test_name, filepath in results.items():
        print(f"  - {test_name}: {filepath}")
    
    return results


if __name__ == "__main__":
    # Example usage
    model_path = "results_cs_wae/cs_wae_model.pth"  # Update this path
    
    if os.path.exists(model_path):
        results = run_advanced_visual_analysis(model_path)
        print("\n✅ Advanced visual analysis completed!")
    else:
        print(f"❌ Model file not found: {model_path}")
        print("Please update the model_path variable or train a model first.")
