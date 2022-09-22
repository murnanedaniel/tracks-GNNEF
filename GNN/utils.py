import os

from torch_geometric.data import Data, Dataset#,DataLoader
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch


class GraphDataset(Dataset):
    def __init__(self,graph_files,transform=None, pre_transform=None):
        super(GraphDataset,self).__init__()

        self.graph_files = graph_files
    
    @property                 
    def raw_file_names(self):
      return self.graph_files

    @property
    def processed_file_names(self):
      return []
        
    def get(self, idx):
      return torch.load(self.graph_files[idx])
          
    def len(self):
      return len(self.graph_files)

def get_pt_files(input_dir, subdir="Train"):
    graph_files_train = []
    for root, dirs, files in os.walk(os.path.join(input_dir, subdir)):
        graph_files_train.extend(os.path.join(root, file) for file in files if file.endswith(".pt"))

    return graph_files_train

def binary_acc(y_pred, y_test,thld):
  """
  returns accuracy based on a given treshold
  """

  # true positives edges with ouput prediction bigger than thld(1) and label = 1
  TP = torch.sum((y_test==1.).squeeze() & 
                           (y_pred>thld).squeeze()).item()
  # true negatives edges with ouput prediction smaller than thld(0) and label = 0
  TN = torch.sum((y_test==0.).squeeze() & 
                           (y_pred<thld).squeeze()).item()
  # False positives edges with ouput prediction bigger than thld(1) and label = 0
  FP = torch.sum((y_test==0.).squeeze() & 
                           (y_pred>thld).squeeze()).item()
  # False negatives edges with ouput prediction smaller than thld(0) and label = 1                     
  FN = torch.sum((y_test==1.).squeeze() & 
                           (y_pred<thld).squeeze()).item() 
  #how many correct predictions are made, if FP = 0 and FN = 0 acc = 1                       
  acc = (TP+TN)/(TP+TN+FP+FN)
    
  return acc

def make_mlp(
    input_size,
    sizes,
    hidden_activation="SilU",
    output_activation=None,
    layer_norm=True,
    batch_norm=False,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1], elementwise_affine=False))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[i + 1], track_running_stats=False, affine=False))
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1], elementwise_affine=False))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[-1], track_running_stats=False, affine=False))
        layers.append(output_activation())
    return nn.Sequential(*layers)