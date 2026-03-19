import torch

class Graph:
    def __init__(self, adj_matrix, pos=None, name="Graphe", width=800, height=600):
        self.N = adj_matrix.shape[0]
        self.adj = adj_matrix.float()
        self.name = name
        self.width = width
        self.height = height
        if pos is None:
            self.pos = torch.rand(self.N, 2) * torch.tensor([width-1, height-1])
            self.pos.requires_grad_(True)
        else:
            self.pos = pos.clone().detach().requires_grad_(True)

    def get_edges(self):
        edges = []
        for i in range(self.N):
            for j in range(i+1, self.N):
                if self.adj[i, j] > 0.5:
                    edges.append((i, j))
        return edges

    def to(self, device):
        self.adj = self.adj.to(device)
        self.pos = self.pos.to(device)
        return self