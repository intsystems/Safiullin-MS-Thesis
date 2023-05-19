from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import matplotlib.pyplot as plt



def train(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for batch_data, batch_targets in dataloader:
        batch_data = batch_data.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        predictions = model(batch_data)
        loss = criterion(predictions, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)
def train_model(model, n_epochs, train_loader, criterion, optimizer, device):
    """
    Train a model for n_epochs using the provided optimizer and criterion.

    Parameters:
    model (torch.nn.Module): The model to be trained.
    n_epochs (int): The number of epochs to train the model.
    train_loader (DataLoader): The DataLoader to use for training.
    criterion: Loss function.
    optimizer: Optimizer.
    device: Device to run the training on.

    Returns:
    torch.nn.Module: The trained model.
    """
    model = model.double().to(device)
    for epoch in range(n_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {train_loss:.6f}')
    return model



        
        
        
def get_predictions(model, target_series, reconstructed_data_test, device):
    """
    Get predictions from the model for the target series.

    Parameters:
    model (torch.nn.Module): The model used to make predictions.
    target_series (np.array): The target time series data.
    reconstructed_data_test (np.array): The test data reconstructed by the model.
    device: Device used for making predictions.

    Returns:
    np.array: The prediction data.
    """
    model.eval()
    with torch.no_grad():
        output_data = model(Variable(torch.tensor(reconstructed_data_test[target_series,:,:]).double().unsqueeze(0)).to(device))
    return output_data[0].cpu().numpy()
def min_max_normalize_data(data):
    """
    Normalize data using min-max normalization.

    Parameters:
    data (np.array): The data to be normalized.

    Returns:
    np.array: The min-max normalized data.
    """
    min_value = np.min(data)
    max_value = np.max(data)
    
    return (data - min_value) / (max_value - min_value)

def plot_data(prediction_data, target_data):
    """
    Plot prediction data against the target data.

    Parameters:
    prediction_data (np.array): The data predicted by the model.
    target_data (np.array): The target data.

    Returns:
    None
    """
    
    
    axis = ['x', 'y', 'z']
    for AXIS in range(3):
        plt.plot(min_max_normalize_data(prediction_data[AXIS]), label='prediction')
        plt.plot(min_max_normalize_data(target_data[AXIS]), label='target')
        plt.legend()
        plt.title(axis[AXIS])
        plt.show()
