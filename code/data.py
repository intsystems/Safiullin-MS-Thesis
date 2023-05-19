from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset




def collect_data(df, num_features, num_channels):
    """
    Collects data in tensor format from the given dataframe and normalizes it.

    Parameters:
    df (pd.DataFrame): Dataframe containing the data.

    Returns:
    tuple: X_tensor (np.array), smart_indexes (list), watch_indexes (list)
    """

    
    X_tensor = np.zeros((2* len(df.UserID.unique()) * len(df.UserID.unique()),3, 200))
    index = 0
    to_select = []
    count = 0
    smart_indexes = []
    watch_indexes = []
    for user in tqdm(df.UserID.unique()):
        for activity in df.Activity.unique():


                if df[(df.UserID==user) & (df.Activity==activity) ].shape[0]==6:
                    count += 1
                    for device in [0,1]:
                        to_select.append([user,activity,device])
                        if device==0:
                                smart_indexes.append(index)
                        else:
                                watch_indexes.append(index)
                        for axi_ind, axis in enumerate(['x','y','z']):


                            X_tensor[index,axi_ind,:] = (df[(df.Device==device)&(df.UserID==user) & (df.Activity==activity) &( 
                                df.Axis==axis)  ].values[0][:200])

                        index+=1



    return X_tensor, smart_indexes, watch_indexes



def prepare_dataset(reconstructed_data_train, X_tensor, BATCH_SIZE):
    """
    Prepare dataset and dataloader for training.
    
    Parameters:
    reconstructed_data_train (np.array): The training data reconstructed by the model.
    X_tensor (np.array): The tensor data.
    
    Returns:
    DataLoader: DataLoader for the training data.
    """
    dataset = TensorDataset(torch.tensor(reconstructed_data_train[1::2, :, :]).double(), torch.tensor(X_tensor[:808*2,:,:][1::2]).double())
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader


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

def normalize_data(data):
    """
    Normalize data by subtracting mean and dividing by standard deviation.

    Parameters:
    data (np.array): The data to be normalized.

    Returns:
    np.array: The normalized data.
    """
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std