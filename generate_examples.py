import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from mnist_model import get_data_transforms, train_model
import os
import time

def denormalize(tensor):
    """Convert normalized tensor back to original scale"""
    return tensor * 0.5 + 0.5

def generate_example_images(num_examples=5, save_dir='example_images'):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set random seed based on current time to get different images each run
    current_seed = int(time.time())
    torch.manual_seed(current_seed)
    np.random.seed(current_seed)
    
    # Define more aggressive augmentations
    augmentation_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomAffine(0, translate=(0.2, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load datasets
    original_dataset = datasets.MNIST('./data', train=True, download=True)
    augmented_dataset = datasets.MNIST('./data', train=True, download=True,
                                     transform=augmentation_transform)
    
    # Train model to make predictions
    model, _ = train_model(epochs=1)
    model.eval()
    
    # Generate examples
    plt.figure(figsize=(15, 3*num_examples))
    
    for i in range(num_examples):
        idx = np.random.randint(len(original_dataset))
        
        # Get original image
        orig_img, label = original_dataset[idx]
        orig_img = np.array(orig_img)
        
        # Get augmented image
        aug_img, _ = augmented_dataset[idx]
        
        # Make prediction using the normalized augmented image
        with torch.no_grad():
            # Use aug_img directly as it's already properly normalized
            pred = model(aug_img.unsqueeze(0))  # Add batch dimension
            pred_label = pred.argmax(dim=1).item()
        
        # Convert augmented image for display
        aug_img_display = denormalize(aug_img).squeeze().numpy()
        
        # Plot
        plt.subplot(num_examples, 3, i*3 + 1)
        plt.imshow(orig_img, cmap='gray')
        plt.title(f'Original (Label: {label})')
        plt.axis('off')
        
        plt.subplot(num_examples, 3, i*3 + 2)
        plt.imshow(aug_img_display, cmap='gray')
        plt.title('Augmented')
        plt.axis('off')
        
        plt.subplot(num_examples, 3, i*3 + 3)
        plt.imshow(aug_img_display, cmap='gray')
        plt.title(f'Predicted: {pred_label}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mnist_examples.png'))
    plt.close()

if __name__ == "__main__":
    generate_example_images() 