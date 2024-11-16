import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from mnist_model import get_data_transforms, train_model
import os

def denormalize(tensor):
    """Convert normalized tensor back to original scale"""
    return tensor * 0.5 + 0.5

def generate_example_images(num_examples=5, save_dir='example_images'):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Define more aggressive augmentations
    augmentation_transform = transforms.Compose([
        transforms.RandomRotation(30),  # Rotate up to 30 degrees
        transforms.RandomAffine(0, translate=(0.2, 0.2)),  # Translate up to 20%
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load dataset with original images (no transforms)
    original_dataset = datasets.MNIST('./data', train=True, download=True)
    
    # Load dataset with augmentation transforms
    augmented_dataset = datasets.MNIST('./data', train=True, download=True,
                                     transform=augmentation_transform)
    
    # Train model to make predictions
    model, _ = train_model(epochs=1)
    model.eval()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate examples
    plt.figure(figsize=(15, 3*num_examples))
    
    for i in range(num_examples):
        idx = np.random.randint(len(original_dataset))
        
        # Get original image
        orig_img, label = original_dataset[idx]
        orig_img = np.array(orig_img)
        
        # Get augmented image
        aug_img, _ = augmented_dataset[idx]
        aug_img = denormalize(aug_img).squeeze().numpy()
        
        # Get prediction
        with torch.no_grad():
            # Convert numpy array to torch tensor
            aug_tensor = torch.from_numpy(aug_img).float()
            pred = model(aug_tensor.unsqueeze(0).unsqueeze(0))
            pred_label = pred.argmax(dim=1).item()
        
        # Plot
        plt.subplot(num_examples, 3, i*3 + 1)
        plt.imshow(orig_img, cmap='gray')
        plt.title(f'Original (Label: {label})')
        plt.axis('off')
        
        plt.subplot(num_examples, 3, i*3 + 2)
        plt.imshow(aug_img, cmap='gray')
        plt.title('Augmented')
        plt.axis('off')
        
        plt.subplot(num_examples, 3, i*3 + 3)
        plt.imshow(aug_img, cmap='gray')
        plt.title(f'Predicted: {pred_label}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mnist_examples.png'))
    plt.close()

if __name__ == "__main__":
    generate_example_images() 