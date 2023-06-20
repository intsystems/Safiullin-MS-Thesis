
import os
import random
import warnings
from itertools import product
from pyriemann.utils.mean import mean_riemann

import cvxpy as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import scipy
import sklearn.feature_selection as sklfs
import torch
import torch.nn as nn
import torch.optim as optim
from pyriemann.estimation import Covariances
from pyriemann.utils import mean as riemann_mean
from pyriemann.utils.tangentspace import tangent_space, untangent_space
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorly.decomposition import tucker
from tensorly.tenalg import mode_dot, multi_mode_dot, kronecker
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from tensorly import decomposition, tenalg
from qpfs import *


import warnings
warnings.filterwarnings("ignore")





def train_riemann_model(tangent_space_data_phone_train, tangent_space_data_watch_train, n_features, hidden_sizes, dropout, learning_rate, n_epochs, batch_size, device):
    '''
    Train a Riemannian Model with provided parameters

    Parameters:
    tangent_space_data_phone_train (torch.Tensor): training data from the phone
    tangent_space_data_watch_train (torch.Tensor): training data from the watch
    n_features (int): number of features in the input data
    hidden_sizes (list): list of hidden layer sizes
    dropout (float): dropout rate for the layers
    learning_rate (float): learning rate for the optimizer
    n_epochs (int): number of epochs for training
    batch_size (int): batch size for training
    device (str): device to train the model on ('cpu' or 'cuda')

    Returns:
    RiemannFullyConnected: trained Riemannian model
    '''
    data_loader_riemann = prepare_dataset(tangent_space_data_phone_train, tangent_space_data_watch_train, batch_size)
    rieman_model = RiemannFullyConnected(n_features=n_features, hidden_sizes=hidden_sizes, dropout=dropout).double().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(rieman_model.parameters(), lr=learning_rate)
    return train_model(rieman_model, data_loader_riemann, n_epochs, criterion, optimizer, device)




def get_correlation_and_mse(y_test, output_data_series):
    correlations = []
    mses = []
    for target, prediction in zip(y_test, output_data_series.cpu().numpy()):
        correlation = np.corrcoef(target, prediction)[0, 1]
        correlations.append(correlation)

        mse = mean_squared_error(target, prediction, squared=False)
        mses.append(mse)
    average_correlation = np.mean(correlations)
    average_mse = np.mean(mses)
    return average_correlation, average_mse

def riem_main(qpfs_feat, qpfs_feat_nums, X_tensor, X_tensor_old, riemann_params, custom_model_params, device):
    '''
    Perform a series of experiments with different number of features. For each experiment, perform cross-validation
    for performance estimation. 

    Args:
        qpfs_feat_nums: list of int
            Numbers of features to use in the experiment.
        cv_indices: list of int
            List of cross-validation indices.
        X_tensor: torch.Tensor
            Tensor of the data.
        X_tensor_old: torch.Tensor
            Tensor of the old data.
        qpfs_feat: float
            The feature selection parameter.
        train_index: int
            Index for separating training and testing data.
        last_index: int
            Last index of data.
        riemann_params: dict
            Parameters for Riemann model training. Includes 'n_features', 'hidden_sizes', 'dropout', 'learning_rate', 'n_epochs', 'batch_size'.
        custom_model_params: dict
            Parameters for custom model training. Includes 'n_ch', 'n_features', 'hidden_sizes', 'dropout', 'learning_rate', 'n_epochs', 'batch_size'.
        device: str
            Device to train the models on (e.g., 'cuda', 'cpu').

    Returns:
        iter_results: list of list
            For each number of features, contains a list with the number of features, correlations, and mean square errors from CV.
    '''
    iter_results = []

        
    for qpfs_feat_num in tqdm(qpfs_feat_nums):
        correlations = []
        mses = []
        for _ in range(5):
            # Data preparation
            tangent_space_data_phone_train, riemann_mean_phone_train, tangent_space_data_watch_train, riemann_mean_watch_train,\
            tangent_space_data_phone_test, tangent_space_data_watch_test, reconstructed_data_train, reconstructed_data_test, X_watch_train, X_watch_test = \
                process_tensors(X_tensor,
                                X_tensor_old,
                                qpfs_feat_num, qpfs_feat)

            # Training Riemann model
            rieman_model = train_riemann_model(tangent_space_data_phone_train, tangent_space_data_watch_train, **riemann_params, device=device)
            output_data = eval_model(rieman_model, Variable(torch.tensor(tangent_space_data_phone_test)), device)

            # Training custom model
            model = train_custom_model(reconstructed_data_train, X_watch_train, **custom_model_params, device=device)
            output_data_series = eval_model(model, torch.tensor(untangent_space(output_data.cpu(), riemann_mean_watch_train)), device)

            # Evaluating models
            average_correlation, average_mse = get_correlation_and_mse(X_watch_test, output_data_series)
            print(qpfs_feat_num, average_correlation, average_mse)

            correlations.append(average_correlation)
            mses.append(average_mse)

        iter_results.append([qpfs_feat_num,correlations,mses])
        print (qpfs_feat_num,correlations,mses)
        
    return iter_results




def process_tensors(X_tensor, X_tensor_old, qpfs_feat_num, qpfs_feat):
    '''
    Process tensors to prepare them for Riemannian and custom model training

    Parameters:
    X_tensor (torch.Tensor): input tensor data
    X_tensor_old (torch.Tensor): old tensor data
    qpfs_feat (torch.Tensor): features selected by QPFS
    qpfs_feat_num (int): number of features selected by QPFS

    Returns:
    tuple: tuple containing:
            - tangent_space_data_phone_train (torch.Tensor): tangent space data for phone for training
            - riemann_mean_phone_train (torch.Tensor): mean Riemannian data for phone for training
            - tangent_space_data_watch_train (torch.Tensor): tangent space data for watch for training
            - riemann_mean_watch_train (torch.Tensor): mean Riemannian data for watch for training
            - tangent_space_data_phone_test (torch.Tensor): tangent space data for phone for testing
            - tangent_space_data_watch_test (torch.Tensor): tangent space data for watch for testing
            - reconstructed_data_train (torch.Tensor): reconstructed data for training
            - reconstructed_data_test (torch.Tensor): reconstructed data for testing
    '''

    X_phone_train_, X_phone_test_, X_watch_train_, X_watch_test_ = train_test_split(X_tensor[::2], X_tensor_old[1::2], test_size=0.2)
    X_phone_train_temp, X_phone_test_temp = X_phone_train_[:,:,:qpfs_feat_num].copy(), X_phone_test_[:,:,:qpfs_feat_num].copy()
            
    qpfs_feat = []
    for axis in range(3):  
            qpfs = QPFS(k=150)
            qpfs.fit(torch.tensor(X_phone_train_[:,axis, :]).float(),
                          torch.tensor(X_watch_train_[:,axis, :]).float())
            qpfs_feat.append(qpfs.get_topk_indices().numpy())
            
                
    if qpfs_feat_num != 0:
            X_phone_train_temp[:, 0, :], X_phone_test_temp[:, 0, :] = X_phone_train_[:,0,qpfs_feat[0][:qpfs_feat_num]], X_phone_test_[:,0,qpfs_feat[0][:qpfs_feat_num]]
                                                                                                                              
            X_phone_train_temp[:, 1, :], X_phone_test_temp[:, 1, :] = X_phone_train_[:,1,qpfs_feat[1][:qpfs_feat_num]], X_phone_test_[:,1,qpfs_feat[1][:qpfs_feat_num]]
                                                                                                                              
            X_phone_train_temp[:, 2, :], X_phone_test_temp[:, 2, :] = X_phone_train_[:,2,qpfs_feat[2][:qpfs_feat_num]], X_phone_test_[:,2,qpfs_feat[2][:qpfs_feat_num]]
            


    else:
            qpfs_feat_num = 200
            X_phone_train_temp, X_phone_test_temp = X_phone_train_, X_phone_test_


    X_phone_train, X_phone_test, X_watch_train, X_watch_test = X_phone_train_temp, X_phone_test_temp, X_watch_train_, X_watch_test_

    
    
    
    tangent_space_data_phone_train, riemann_mean_phone_train = to_riemann_space(X_phone_train)
    tangent_space_data_watch_train, riemann_mean_watch_train = to_riemann_space(X_watch_train)
    
    cov_matrices_phone = Covariances().fit_transform(X_phone_test)
    tangent_space_data_phone_test = tangent_space(cov_matrices_phone, riemann_mean_phone_train)

    cov_matrices_watch = Covariances().fit_transform(X_watch_test)
    tangent_space_data_watch_test = tangent_space(cov_matrices_watch, riemann_mean_watch_train)
    
    
    reconstructed_data_train = untangent_space(tangent_space_data_watch_train, riemann_mean_watch_train)

    reconstructed_data_test = untangent_space(tangent_space_data_watch_test, riemann_mean_watch_train)

    return tangent_space_data_phone_train, riemann_mean_phone_train, tangent_space_data_watch_train, riemann_mean_watch_train, tangent_space_data_phone_test, tangent_space_data_watch_test, reconstructed_data_train, reconstructed_data_test, X_watch_train, X_watch_test


def train_custom_model(reconstructed_data_train, y, n_ch, n_features, hidden_sizes, dropout, learning_rate, n_epochs, batch_size, device):
    '''
    Train a Custom Model with provided parameters

    Parameters:
    reconstructed_data_train (torch.Tensor): training data after reconstruction
    X_tensor_old (torch.Tensor): original tensor data
    cv_ind (int): index for cross validation
    train_index (int): index for the training data
    n_ch (int): number of channels in the input data
    n_features (int): number of features in the input data
    hidden_sizes (list): list of hidden layer sizes
    dropout (float): dropout rate for the layers
    learning_rate (float): learning rate for the optimizer
    n_epochs (int): number of epochs for training
    batch_size (int): batch size for training
    device (str): device to train the model on ('cpu' or 'cuda')

    Returns:
    CustomFullyConnected: trained custom model
    '''
    
    train_loader = prepare_dataset(reconstructed_data_train, y, batch_size)
    criterion = nn.MSELoss()
    model = CustomFullyConnected(n_ch=n_ch, n_features=n_features, hidden_sizes=hidden_sizes, dropout=dropout).double().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return train_model(model, train_loader, n_epochs, criterion, optimizer, device)
def eval_model(model, input_data, device):
    '''
    Evaluate a model on input data

    Parameters:
    model (nn.Module): model to evaluate
    input_data (torch.Tensor): input data for evaluation
    device (str): device to evaluate the model on ('cpu' or 'cuda')

    Returns:
    torch.Tensor: output data from the model
    '''
    with torch.no_grad():
        output_data = model(Variable(input_data).double().to(device))
    return output_data



def to_riemann_space(data):
    cov_matrices = Covariances().fit_transform(data)
    riemann_mean = mean_riemann(cov_matrices)
    tangent_space_data = tangent_space(cov_matrices, riemann_mean)
    return tangent_space_data, riemann_mean
def to_time_series(tangent_space_data, riemann_mean):
    cov_matrices = untangent_space(tangent_space_data, riemann_mean)
    return cov_matrices


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


def train_model(model, train_loader, n_epochs, criterion, optimizer, device):
    for epoch in range(n_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
    model.eval()
    return model
    

class TimeseriesDataset(torch.utils.data.Dataset):   
    def __init__(self, X, y):
        self.X = X
        self.y = y
#         self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() 

    def __getitem__(self, index):
        return ((self.X[index,:]), (self.y[index]))
    
def train_autoencoder(model, data_loader, num_epochs=100):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=7e-4)
    for epoch in range(num_epochs):
        for (img, _) in data_loader:
            img = img.cuda()
            recon = model(img)
            loss = criterion(recon, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def encode_data(model, data_loader):
    encoded_data = []
    for (img, _) in data_loader:
        batch_encoded = model.encoder(img.cuda()).cpu().detach().numpy()
        encoded_data.extend(batch_encoded)
    return np.array(encoded_data).reshape(-1, 16)

def train_regression(X_train, y_train, num_components=16):
    return PLSRegression(n_components=num_components).fit(X_train, y_train)

def train_multi_net(net, X_train, y_train, num_epochs=1000):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
    inputs = Variable(torch.Tensor(X_train)).cuda()
    outputs = Variable(torch.Tensor(y_train)).cuda()
    for _ in range(num_epochs):
        prediction = net(inputs)
        loss = criterion(prediction, outputs) 
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()
    return net    




class CustomFullyConnected(nn.Module):
    def __init__(self, n_ch, n_features, hidden_sizes, dropout=None):
        super(CustomFullyConnected, self).__init__()
        layers = []
        input_size = n_ch * n_ch
        self.n_ch = n_ch
        self.n_features = n_features 
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(p=dropout))
            input_size = hidden_size
            
        layers.append(nn.Linear(input_size, n_ch * n_features))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.network(x)
        x = x.view(x.size(0), self.n_ch, self.n_features)
        return x

class RiemannFullyConnected(nn.Module):
    def __init__(self, n_features, hidden_sizes, dropout=None):
        super().__init__()
        layers = []
        input_size = n_features
        self.n_features = n_features
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(p=dropout))
            input_size = hidden_size
            
        layers.append(nn.Linear(input_size, n_features))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.network(x)
        x = x.view(x.size(0), self.n_features)
        return x



def prepare_dataset(reconstructed_data_train, X_tensor, BATCH_SIZE):
    """
    Prepare dataset and dataloader for training.
    
    Parameters:
    reconstructed_data_train (np.array): The training data reconstructed by the model.
    X_tensor (np.array): The tensor data.
    
    Returns:
    DataLoader: DataLoader for the training data.
    """
    dataset = TensorDataset(torch.tensor(reconstructed_data_train).double(), torch.tensor(X_tensor).double())
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader
    

    
    
class Autoencoder_Linear(nn.Module):
    def __init__(self, n_feat):
        super().__init__()        
        self.n_feat = n_feat

        self.encoder = nn.Sequential(
            nn.Linear(self.n_feat*3, 32),
            nn.ReLU(),
            nn.Linear(32, 16) 
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_feat*3)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class Autoencoder_Linear_watch(nn.Module):
    def __init__(self):
        super().__init__()        
        self.encoder = nn.Sequential(
            nn.Linear(200*3, 64), 
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 16) 
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 200*3)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

class model_multi(nn.Module):
    def __init__(self):
        super(model_multi, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 16),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    



    
def flatten_tensor(input_tensor):
    """
    Function to flatten a tensor from (1, 3, 200) to (1, 3*200)

    Args:
    input_tensor (torch.Tensor): input tensor of shape (1, 3, 200)

    Returns:
    torch.Tensor: flattened tensor of shape (1, 3*200)
    """
    # Get the size of the tensor
    size = input_tensor.size()

    # Reshape the tensor and return
    return input_tensor.view(size[0], -1)


def unflatten_tensor(flattened_tensor):
    """
    Function to convert a flattened tensor back to its original shape

    Args:
    flattened_tensor (torch.Tensor): flattened tensor of shape (1, 3*200)

    Returns:
    torch.Tensor: original tensor of shape (1, 3, 200)
    """
    # Get the size of the tensor
    size = flattened_tensor.size()

    # Reshape the tensor and return
    return flattened_tensor.view(size[0], 3, -1).numpy()

def aeCrossVal(qpfs_feat, qpfs_feat_nums, X_tensor, X_tensor_old,smart_indexes, watch_indexes, device):
    '''
    Perform a series of experiments with different number of features. For each experiment, perform cross-validation
    for performance estimation.

    Args:
        qpfs_feat_nums: list of int
            Numbers of features to use in the experiment.
        X_tensor: np.array
            The data array.
        X_tensor_old: np.array
            The data array.
        smart_indexes: list of int
            Indices of the smartwatch data.
        watch_indexes: list of int
            Indices of the watch data.
        qpfs_2d_feat: np.array
            The 2D feature selection parameter.
        kf_params: dict
            Parameters for the K-Fold split. Includes 'n_splits', 'shuffle', 'random_state'.
        device: str
            Device to train the models on (e.g., 'cuda', 'cpu').

    Returns:
        iter_results: list of list
            For each number of features, contains a list with the number of features, correlations, and mean square errors.
    '''
    iter_results = []
    for qpfs_feat_num in tqdm(qpfs_feat_nums):
        correlations = []
        mses = []
        # Prepare data
        
        
        
        X_phone = torch.from_numpy(np.array(X_tensor[smart_indexes[:-1], :, :], dtype = np.float32))
        X_watch = torch.from_numpy(np.array(X_tensor[watch_indexes[:-1], :, :], dtype = np.float32))
        
  

            
        
        

        # Set up K-Fold cross validation
        
        correlations_cv = []
        mses_cv = []
        # Iterate over each fold
        for _ in range(5):
            # Split data into training and test sets
            X_phone_train_, X_phone_test_, X_watch_train_, X_watch_test_ = train_test_split(X_phone, X_watch, test_size=0.2)
            X_phone_train_temp, X_phone_test_temp = X_phone_train_[:,:,:qpfs_feat_num].clone(), X_phone_test_[:,:,:qpfs_feat_num].clone()
            
            qpfs_feat = []
            for axis in range(3):  
                qpfs = QPFS(k=150)
                qpfs.fit(torch.tensor(X_phone_train_[:,axis, :]).float(),
                          torch.tensor(X_watch_train_[:,axis, :]).float())
                qpfs_feat.append(qpfs.get_topk_indices().numpy())
                
            if qpfs_feat_num != 0:
                X_phone_train_temp[:, 0, :], X_phone_test_temp[:, 0, :] = X_phone_train_[:,0,qpfs_feat[0][:qpfs_feat_num]], X_phone_test_[:,0,qpfs_feat[0][:qpfs_feat_num]]
                                                                                                                              
                X_phone_train_temp[:, 1, :], X_phone_test_temp[:, 1, :] = X_phone_train_[:,1,qpfs_feat[1][:qpfs_feat_num]], X_phone_test_[:,1,qpfs_feat[1][:qpfs_feat_num]]
                                                                                                                              
                X_phone_train_temp[:, 2, :], X_phone_test_temp[:, 2, :] = X_phone_train_[:,2,qpfs_feat[2][:qpfs_feat_num]], X_phone_test_[:,2,qpfs_feat[2][:qpfs_feat_num]]
            


            else:
                qpfs_feat_num = 200
                X_phone_train_temp, X_phone_test_temp = X_phone_train_, X_watch_train_

            X_phone_train, X_phone_test, X_watch_train, X_watch_test = flatten_tensor(X_phone_train_temp), flatten_tensor(X_phone_test_temp), flatten_tensor(X_watch_train_), flatten_tensor(X_watch_test_)
            
            model_phone = Autoencoder_Linear(n_feat=qpfs_feat_num).cuda()
            model_watch = Autoencoder_Linear_watch().cuda()
            net = model_multi().cuda()



            data_loader_phone_train = DataLoader(TimeseriesDataset(X_phone_train, np.arange((len(X_phone_train)))), batch_size=256, shuffle=True)
            data_loader_phone_test = DataLoader(TimeseriesDataset(X_phone_test, np.arange((len(X_phone_test)))), batch_size=256, shuffle=True)
            data_loader_watch_train = DataLoader(TimeseriesDataset(X_watch_train, np.arange((len(X_phone_train)))), batch_size=256, shuffle=True)
            data_loader_watch_test = DataLoader(TimeseriesDataset(X_watch_test, np.arange((len(X_phone_test)))), batch_size=256, shuffle=True)



            # Train Autoencoders
            model_phone = train_autoencoder(model_phone, data_loader_phone_train)
            model_watch = train_autoencoder(model_watch, data_loader_watch_train)

            # Encode data using trained Autoencoders
            encoded_from_phone_train = encode_data(model_phone, data_loader_phone_train)
            encoded_from_watch_train = encode_data(model_watch, data_loader_watch_train)

            # Train regression
            pls = train_regression(encoded_from_phone_train, encoded_from_watch_train)

            # Encode test data using trained Autoencoders
            encoded_from_phone_test = encode_data(model_phone, data_loader_phone_test)

            # Predict with regression
            y_pred = pls.predict(encoded_from_phone_test)

            # Train multi_net
            net = train_multi_net(net, encoded_from_phone_train, encoded_from_watch_train)
            net.eval()
            with torch.no_grad():
                output_data_series_net = net(Variable(torch.tensor(encoded_from_phone_test)).float().to(device))
            
            
            
     
            y_pred_orig_dim_pls = unflatten_tensor(model_watch.decoder(torch.tensor(y_pred).float().cuda()).cpu().detach())


            # Convert y_pred back to original dimension
            y_pred_orig_dim = unflatten_tensor(model_watch.decoder(output_data_series_net.cuda()).cpu().detach())
            correlations = []
            mses = []
            for target, prediction in zip(X_watch_test_, y_pred_orig_dim):

                correlation = np.corrcoef(target, prediction)[0, 1]
                correlations.append(correlation)
                
                mse = mean_squared_error(target, prediction, squared=False)
                mses.append(mse)
                
            average_correlation = np.mean(correlations)
            average_mse = np.mean(mses)
            
            correlations_cv.append(['NN',average_correlation])
            mses_cv.append(['NN', average_mse])
    
            
            correlations = []
            mses = []
            for target, prediction in zip(X_watch_test_, y_pred_orig_dim_pls):

                correlation = np.corrcoef(target, prediction)[0, 1]
                correlations.append(correlation)
                
                mse = mean_squared_error(target, prediction, squared=False)
                mses.append(mse)
                
            average_correlation = np.mean(correlations)
            average_mse = np.mean(mses)
            
            correlations_cv.append(['pls',average_correlation])
            mses_cv.append(['pls', average_mse])
   
        
            average_correlation = np.mean(correlations)
            average_mse = np.mean(mses)
            
            if qpfs_feat_num==200:
                qpfs_feat_num=0
                  
        iter_results.append([qpfs_feat_num,correlations_cv,mses_cv])
    return iter_results

