import torch 
import numpy as np
import sklearn.feature_selection as sklfs
import scipy as sc
import cvxpy as cvx
from sklearn.preprocessing import MinMaxScaler, StandardScaler



def get_corr_matrix(X, Y=None, fill=0):
    if Y is None:
        Y = X
    if len(Y.shape) == 1:
        Y = torch.unsqueeze(Y, dim  = 1)
    if len(X.shape) == 1:
        X = torch.unsqueeze(X, dim = 1)
    
    X_ = (X - X.mean(dim=0))
    Y_ = (Y - Y.mean(dim=0))
    
    idxs_nz_x = torch.where(torch.sum(X_ ** 2, dim = 0) != 0)[0]
    idxs_nz_y = torch.where(torch.sum(Y_ ** 2, dim = 0) != 0)[0]
    X_ = X_[:, idxs_nz_x]
    Y_ = Y_[:, idxs_nz_y]
    corr = torch.ones((X.shape[1], Y.shape[1])) * fill
    for i, x in enumerate(X_.T):
        corr[idxs_nz_x[i], idxs_nz_y] = (Y_.T@ x) / torch.sqrt((x ** 2).sum() * (Y_ ** 2).sum(dim=0, keepdim=True))
    return corr


def shift_spectrum(Q, eps=0.):
    lamb_min = sc.linalg.eigh(Q)[0][0]
    if lamb_min < 0:
        Q = Q - (lamb_min - eps) * torch.eye(*Q.shape)
    return Q, lamb_min


class QPFS:
    def __init__(self, sim='corr', k  = 10):
        if sim not in ['corr', 'info']:
            raise ValueError('Similarity measure should be "corr" or "info"')
        self.sim = sim
        self.n_features = k
    
    def get_params(self, X, y):
        if self.sim == 'corr':
            self.Q = torch.abs(get_corr_matrix(X, fill=1))
            self.b = torch.unsqueeze(torch.sum(torch.abs(get_corr_matrix(X, y)), dim=1),1)
#             print (self.b)
        elif self.sim == 'info':
            self.Q = torch.ones([X.shape[1], X.shape[1]])
            self.b = torch.zeros((X.shape[1], 1))
            for j in range(self.n_features):
                self.Q[:, j] = torch.tensor(sklfs.mutual_info_regression((X), (X[:, j])))
            if len(y.shape) == 1:
                self.b = torch.unsqueeze(torch.tensor(sklfs.mutual_info_regression(X, y)), dim = 1)
            else:
                for y_ in y:
                    self.b += torch.tensor(sklfs.mutual_info_regression(X, y_))
        self.n = self.Q.shape[0]
    
    def get_alpha(self):
        return self.Q.mean() / (self.Q.mean() + self.b.mean())

    def fit(self, X, y):
        self.get_params(X, y)
        alpha = self.get_alpha()
        self.solve_problem(alpha)
    
    def solve_problem(self, alpha):
        
        c = torch.ones((self.n, 1))
        
        Q, _ = shift_spectrum(self.Q)
        
        x = cvx.Variable(self.n)
        objective = cvx.Minimize((1 - alpha) * cvx.quad_form(x, Q) - 
                                 alpha * self.b.T * x)
        constraints = [x >= 0, c.T * x == 1]
        prob = cvx.Problem(objective, constraints)

        prob.solve()

        self.status = prob.status
        self.score = torch.tensor(x.value).flatten()
        
    def get_topk_indices(self):
        return torch.argsort(self.score).flip(dims = [0])[:self.n_features]