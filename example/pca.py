import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from scipy.integrate import quad

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v

class PCA(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_


def read_csv_from_folder(folder_path, B=0, K=500):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    print(len(files))
    df_list = []
    
    for i, file in enumerate(files):
        if i < B: continue
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df_list.append(df)
        if i >= K:
            break
    return df_list

def read_line(data, l):
    o = []
    for d in data:
        o.append(d.iloc[l])
    return o

# path_1 = "./safe_13b/"
# path_2 = "./unsafe_13b/"

path_1 = "./safe_hs/"
path_2 = "./unsafe_hs/"

data_neg = read_csv_from_folder(path_1, 0, 300)
data_pos = read_csv_from_folder(path_2, 0, 300)

custom_legend = [
    Patch(facecolor='dodgerblue', label='Unsafe representation'),
    Patch(facecolor='firebrick', label='Safe representation')
]

####################################################################

def kl_divergence(p, q, x_range, y_range):

    x = np.linspace(x_range[0], x_range[1], 200)
    y = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    p_values = p(positions)
    q_values = q(positions)
    

    kl = p_values * np.log(p_values / q_values)
    kl[np.isnan(kl) | np.isinf(kl)] = 0  # 
    return np.sum(kl) * (x[1] - x[0]) * (y[1] - y[0])

def kde(x1,x2,y1,y2):

    x_range = (min(np.min(x1), np.min(x2)), max(np.max(x1), np.max(x2)))
    y_range = (min(np.min(y1), np.min(y2)), max(np.max(y1), np.max(y2)))
    kde1 = gaussian_kde(np.vstack([x1, y1]))
    kde2 = gaussian_kde(np.vstack([x2, y2]))

    kl_pq = kl_divergence(kde1, kde2, x_range, y_range)
    kl_qp = kl_divergence(kde2, kde1, x_range, y_range)
    return kl_pq,kl_qp


##################################################################

def showw(line):
    s_neg = read_line(data_neg, line)
    s_pos = read_line(data_pos, line)
    # com = pd.concat([s_neg, s_pos], ignore_index=True)

    com = s_neg + s_pos
    com = torch.tensor(com)
    
    s_neg = torch.tensor(s_neg)
    s_pos = torch.tensor(s_pos)

    pca = PCA(n_components=2)
    # pca_neg = PCA(n_components=1)
    # pca_pos = PCA(n_components=1)

    pca.fit(com)

    
    
    neg = pca(s_neg)
    pos = pca(s_pos)

    neg_x = [row[0] for row in neg]
    neg_y = [row[1] for row in neg]
    pos_x = [row[0] for row in pos]
    pos_y = [row[1] for row in pos]

    neg_x = np.array(neg_x)
    neg_y = np.array(neg_y)
    pos_x = np.array(pos_x)
    pos_y = np.array(pos_y)

    
    # print(neg_x)
    # print(neg_y)


    # plt.figure(figsize=(6, 6))

    fig, ax = plt.subplots(figsize=(6, 6))
    p1 = sns.kdeplot(x=neg_x, y=neg_y, cmap="Blues", fill=False, alpha=1.0,  label='Unsafe', ax=ax,)
    p2 = sns.kdeplot(x=pos_x, y=pos_y, cmap="Reds", fill=False, alpha=1.0,  label='Safe', ax=ax,)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    labels = ['Unsafe', 'Safe']
    # handles, labels = ax.get_legend_handles_labels()
    # print(labels)
    # ax.legend(labels)
    center1_x, center1_y = np.mean(neg_x), np.mean(neg_y)
    center2_x, center2_y = np.mean(pos_x), np.mean(pos_y)

    midpoint_x = (center1_x + center2_x) / 2
    midpoint_y = (center1_y + center2_y) / 2

    slope = (center2_y - center1_y) / (center2_x - center1_x)
    perpendicular_slope = -1 / slope

    length = 0.1  
   
    delta_x = length / np.sqrt(1 + perpendicular_slope**2)
    delta_y = perpendicular_slope * delta_x

    x_vals = [midpoint_x - delta_x, midpoint_x + delta_x]
    y_vals = [midpoint_y - delta_y, midpoint_y + delta_y]

    kl1,kl2 = kde(neg_x,neg_y,pos_x,pos_y)
    ax.set_title(f'KL: {kl1:.2f} | {kl2:.2f}', fontsize=24)
    ax.legend(handles=custom_legend,fontsize=20)


    ax.set_xlabel('Principle Component 1',fontsize=16)
    ax.set_ylabel('Principle Component 2',fontsize=16)
    # 
    plt.savefig('./PCA_7b_{}.pdf'.format(line), dpi=300, bbox_inches='tight',format='pdf')
    # list = [1,5,9,13,20,24,28,32]
for i in range(41):
    if i == 0: continue
    showw(i)

