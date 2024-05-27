import matplotlib.pyplot as plt

def save_image_torch(tensor, name="debug"):
    """Saves a torch tensor representing an image into disk, useful for debugging"""
    plt.imsave(f"{name}.png", tensor.detach().cpu().squeeze().permute(1,2,0).numpy())


def save_mask_torch(tensor, name="debug"):
    """Saves a torch tensor representing a mask into disk, useful for debugging"""
    plt.imsave(f"{name}_mask.png", tensor.detach().cpu().squeeze().numpy())