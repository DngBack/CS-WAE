import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def slerp(p0, p1, t, epsilon=1e-8):
    """
    Spherical Linear Interpolation (Slerp).
    p0, p1: start and end vectors (normalized).
    t: a value or tensor containing values from 0 to 1.
    """
    # Calculate the angle between two vectors
    omega = torch.acos(torch.dot(p0, p1).clamp(-1, 1))
    sin_omega = torch.sin(omega)

    # If the two vectors are too close, return the initial vector to avoid division by zero
    if sin_omega.item() < epsilon:
        # Expand p0 to have the same dimensions as the expected output when t is a tensor
        return p0.unsqueeze(0).expand(len(t), -1)

    # Ensure t is on the same device as the vectors
    t = t.to(p0.device)

    # Slerp formula
    a = torch.sin((1.0 - t) * omega) / sin_omega
    b = torch.sin(t * omega) / sin_omega

    # unsqueeze() to perform broadcasting correctly
    return a.unsqueeze(-1) * p0.unsqueeze(0) + b.unsqueeze(-1) * p1.unsqueeze(0)


def plot_slerp(model, save_dir=".", num_steps=10, DEVICE='cuda'):
    """
    Plot and save images generated from Slerp interpolation between prior pairs.
    """
    print("Starting Slerp interpolation image generation...")
    model.eval()

    # Select some interesting digit pairs to interpolate
    interpolation_pairs = [(1, 7), (3, 5), (2, 8), (4, 9)]
    num_pairs = len(interpolation_pairs)

    fig, axes = plt.subplots(num_pairs, num_steps, figsize=(num_steps * 1.5, num_pairs * 1.5))

    with torch.no_grad():
        t_values = torch.linspace(0, 1, num_steps)

        for row, (digit_a, digit_b) in enumerate(interpolation_pairs):
            # GET VECTOR AND FIX BUG: Add `dim=0`
            mu_a = F.normalize(model.prior_mus[digit_a].detach(), dim=0)
            mu_b = F.normalize(model.prior_mus[digit_b].detach(), dim=0)

            # Perform Slerp
            interpolated_z = slerp(mu_a, mu_b, t_values).to(DEVICE)

            # Decode the intermediate z vectors
            generated_images = model.decoder(interpolated_z)

            # Plot the images
            for col, img in enumerate(generated_images):
                ax = axes[row, col]
                ax.imshow(img.cpu().squeeze(), cmap='gray')
                ax.axis('off')

                # Label the first and last images
                if col == 0:
                    ax.set_title(f'{digit_a}')
                if col == num_steps - 1:
                    ax.set_title(f'{digit_b}')

    plt.suptitle("Slerp Interpolation between Priors")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout so title is not overlapped
    plt.savefig(f'{save_dir}/slerp_interpolation.png')
    plt.close(fig)
    print(f"Slerp image saved to '{save_dir}/slerp_interpolation.png'")