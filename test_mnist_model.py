import pytest
import torch
from mnist_model import MnistCNN, train_model

def test_model_parameters_and_accuracy():
    # Train model with more epochs to achieve higher accuracy
    model, accuracy = train_model(epochs=1)
    
    # Check total parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, which exceeds the limit of 25000"
    
    # Check accuracy
    assert accuracy > 95.0, f"Model accuracy {accuracy:.2f}% is below the required 95%"

def test_model_architecture():
    model = MnistCNN()
    
    # Test input shape
    batch_size = 64
    test_input = torch.randn(batch_size, 1, 28, 28)
    output = model(test_input)
    
    # Check output shape
    assert output.shape == (batch_size, 10), f"Expected output shape (64, 10), got {output.shape}"
    
    # Verify parameter count
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, which exceeds the limit of 25000"

def test_data_augmentation():
    """Test if data augmentation is being performed"""
    from mnist_model import get_data_transforms
    
    transforms = get_data_transforms()
    assert len(transforms.transforms) > 1, "No data augmentation transforms found in the pipeline"

def test_train_test_performance():
    """Test if model performs well on both training and test sets"""
    model, train_accuracy, test_accuracy = train_model(epochs=1, return_both_accuracies=True)
    
    # Check both accuracies are above threshold
    assert train_accuracy > 95.0, f"Training accuracy {train_accuracy:.2f}% is below the required 95%"
    assert test_accuracy > 95.0, f"Test accuracy {test_accuracy:.2f}% is below the required 95%"

def test_overfitting_gap():
    """Test if the gap between training and test accuracy is not too large"""
    model, train_accuracy, test_accuracy = train_model(epochs=1, return_both_accuracies=True)
    
    accuracy_gap = abs(train_accuracy - test_accuracy)
    assert accuracy_gap <= 5.0, f"Gap between train ({train_accuracy:.2f}%) and test ({test_accuracy:.2f}%) accuracy is {accuracy_gap:.2f}%, which exceeds the maximum allowed gap of 5%"
    