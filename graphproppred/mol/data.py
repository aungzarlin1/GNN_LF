import torch
from torch_geometric.data import Dataset

class GraphwithEigenEdge(Dataset):
    def __init__(self, graphs, method):
        super().__init__()
        self.graphs = graphs
        self.method = method

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        data = self.graphs[idx]
        extra_feat_path = f"img/molhiv/{self.method}/graph_{idx:05d}.pt"
        extra_feat = torch.load(extra_feat_path)  # Load extra features

        if extra_feat.size(1) > 100:
            extra_feat = extra_feat[:, :100]
            extra_feat = extra_feat / extra_feat.norm(dim=1,keepdim=True)
        elif extra_feat.size(1) < 100:
            extra_feat = extra_feat / extra_feat.norm(dim=1,keepdim=True)
            pad_size = 100 - extra_feat.size(1)
            pad = torch.zeros(extra_feat.size(0), pad_size)
            extra_feat = torch.cat([extra_feat, pad], dim=1)  # pad   

        data.edge_eigen_feat = extra_feat
        return data
