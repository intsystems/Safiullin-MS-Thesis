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

    
    X_tensor = np.zeros((int(len(df)/3),3, 200))
    index = 0
    to_select = []
    count = 0
    smart_indexes = []
    watch_indexes = []
    for user in tqdm(df.UserID.unique()):
        for activity in df.Activity.unique():


                if (df[(df.UserID==user) & (df.Activity==activity) ].shape[0] % 6==0) &(df[(df.UserID==user) & (df.Activity==activity) ].shape[0] // 6>=14):
                    for _ in range(int(df[(df.UserID==user) & (df.Activity==activity) ].shape[0] // 6)):

                            count += 1
                            
                            for device in [0,1]:
                                to_select.append([user,activity,device])
                                if device==0:
                                        smart_indexes.append(index)
                                else:
                                        watch_indexes.append(index)
                                
                                
                                for axi_ind, axis in enumerate(['x','y','z']):
                                        row = df[(df.Device==device)&(df.UserID==user) & (df.Activity==activity) &( 
                                            df.Axis==axis)  ]
                                        if row.shape==0:
                                            print(row.shape,index,axis, user, activity, device)


                                        X_tensor[index,axi_ind,:] = (row.values[0][:200])

                                index+=1



    return X_tensor, smart_indexes, watch_indexes



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