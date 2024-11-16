import pytest
import torch
from mnist_model import MnistCNN, train_model

def test_model_parameters_and_accuracy():
    # Train model with more epochs to achieve higher accuracy
    model, accuracy = train_model(epochs=3)
    
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