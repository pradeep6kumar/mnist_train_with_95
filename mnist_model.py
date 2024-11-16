import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        # Minimal but effective architecture
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 8 filters only
        self.conv2 = nn.Conv2d(8, 28, kernel_size=3, padding=1)  # 16 filters
        self.fc1 = nn.Linear(28 * 7 * 7, 16)  # Small dense layer
        
        self.fc2 = nn.Linear(16, 10)
        #self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 28 * 7 * 7)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_model(epochs=1, batch_size=96):
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model, optimizer and loss function
    model = MnistCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        correct_train = 0
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct_train += pred.eq(target.view_as(pred)).sum().item()
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # Calculate and print training metrics
        train_loss /= len(train_loader)
        train_accuracy = 100. * correct_train / len(train_loader.dataset)
        print(f'\nTraining set: Average loss: {train_loss:.4f}, '
              f'Accuracy: {correct_train}/{len(train_loader.dataset)} ({train_accuracy:.2f}%)')

        # Validation after each epoch
        model.eval()
        correct = 0
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    return model, accuracy

if __name__ == "__main__":
    # Train the model with default 1 epoch
    model, accuracy = train_model(epochs=1)
    
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nTotal number of parameters: {total_params}') 