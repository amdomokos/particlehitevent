import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def initialize_weights(m):
    """Initialize model weights."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def calculate_accuracy(predictions, targets, threshold=0.1):
    """Calculate accuracy based on a threshold."""
    # Define accuracy as prediction within threshold of target
    with torch.no_grad():
        diff = torch.abs(predictions - targets)
        correct = (diff <= threshold).float().mean(dim=1)
        return correct.mean().item()

def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001, max_grad_norm=1.0, device="cuda"):
    """Train the model and return training metrics."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Initialize model weights
    model.apply(initialize_weights)
    
    # Lists to store metrics
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        print(f"Training epoch {epoch+1}...")
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        batches = 0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_accuracy += calculate_accuracy(predictions, batch_y)
            batches += 1
        
        avg_train_loss = epoch_loss / batches
        avg_train_accuracy = epoch_accuracy / batches
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        
        # Testing phase
        print(f"Testing epoch {epoch+1}...")
        test_loss, test_accuracy = test_model(model, test_loader, device, return_accuracy=True)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        # Update learning rate
        scheduler.step(test_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        # Save checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': test_loss,
            }, f"results/checkpoint_epoch_{epoch+1}.pt")
    
    return train_losses, test_losses, train_accuracies, test_accuracies

def test_model(model, test_loader, device="cuda", return_accuracy=False):
    """Test the model and return test metrics."""
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0.0
    test_accuracy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            
            test_loss += loss.item()
            
            if return_accuracy:
                test_accuracy += calculate_accuracy(predictions, targets)
            
            num_batches += 1
    
    avg_test_loss = test_loss / num_batches
    print(f"Test loss: {avg_test_loss:.4f}")

    if return_accuracy:
        avg_test_accuracy = test_accuracy / num_batches
        print(f"Test accuracy: {avg_test_accuracy:.4f}")
        return avg_test_loss, avg_test_accuracy
    
    return avg_test_loss

def save_predictions(model, data_loader, output_file, device="cuda"):
    """Save model predictions to a CSV file."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Generating predictions"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            predictions = model(inputs)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Create DataFrame
    results = pd.DataFrame()
    
    # Add targets
    for i in range(all_targets[0].shape[0]):
        results[f'target_{i}'] = [t[i] for t in all_targets]
    
    # Add predictions
    for i in range(all_predictions[0].shape[0]):
        results[f'prediction_{i}'] = [p[i] for p in all_predictions]
    
    # Add error
    for i in range(all_predictions[0].shape[0]):
        results[f'error_{i}'] = np.abs(results[f'prediction_{i}'] - results[f'target_{i}'])
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    metrics = pd.DataFrame({
        "epoch": list(range(1, num_epochs + 1)),
        "train_loss": train_losses,
        "test_loss": test_losses,
        "train_accuracy": train_accuracies,
        "test_accuracy": test_accuracies,
    })
    metrics.to_csv("results/training_metrics.csv", index=False)

