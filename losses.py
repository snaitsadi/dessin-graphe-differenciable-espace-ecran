import torch
import numpy as np

class GraphLayoutLosses:
    def __init__(self, renderer, scale=50.0):
        self.renderer = renderer
        self.scale = scale

    def stress_loss(self, pos, adj):
        N = pos.shape[0]
        adj_np = adj.detach().cpu().numpy()
        dist_graph = np.full((N, N), np.inf)
        np.fill_diagonal(dist_graph, 0)
        for i in range(N):
            for j in range(N):
                if adj_np[i, j] > 0.5:
                    dist_graph[i, j] = 1
        for k in range(N):
            for i in range(N):
                for j in range(N):
                    if dist_graph[i, k] + dist_graph[k, j] < dist_graph[i, j]:
                        dist_graph[i, j] = dist_graph[i, k] + dist_graph[k, j]
        dist_graph = torch.tensor(dist_graph, dtype=torch.float32, device=pos.device)
        max_dist = torch.max(dist_graph[torch.isfinite(dist_graph)])
        dist_graph[~torch.isfinite(dist_graph)] = max_dist * 2

        pos_i = pos.unsqueeze(1)
        pos_j = pos.unsqueeze(0)
        dist_eucl = torch.sqrt(((pos_i - pos_j)**2).sum(dim=2) + 1e-8)

        weight = 1.0 / (dist_graph + 1e-8)
        loss = weight * (dist_eucl - self.scale * dist_graph)**2
        return loss.mean() * 100

    def node_overlap_loss(self, pos):
        node_img = self.renderer.render_nodes(pos)
        return (node_img**2).mean() * 10

    def edge_crossing_penalty(self, pos, edges):
        if len(edges) == 0:
            return torch.tensor(0.0, device=pos.device)
        edge_img = self.renderer.render_edges(pos, edges)
        laplacian = torch.zeros_like(edge_img)
        laplacian[1:-1, 1:-1] = (edge_img[1:-1, 1:-1] * -4 +
                                  edge_img[:-2, 1:-1] + edge_img[2:, 1:-1] +
                                  edge_img[1:-1, :-2] + edge_img[1:-1, 2:])
        return torch.abs(laplacian).mean() * 100

    def boundary_penalty(self, pos, margin=10):
        dist_left = pos[:, 0]
        dist_right = self.renderer.W - 1 - pos[:, 0]
        dist_bottom = pos[:, 1]
        dist_top = self.renderer.H - 1 - pos[:, 1]
        min_dist = torch.min(torch.stack([dist_left, dist_right, dist_bottom, dist_top], dim=1), dim=1)[0]
        penalty = torch.clamp(margin - min_dist, min=0)
        return penalty.mean() * 10

    def total_loss(self, pos, adj, edges, weights=(1.0, 0.5, 3.0, 0.1)):
        loss_stress = self.stress_loss(pos, adj) * weights[0]
        loss_overlap = self.node_overlap_loss(pos) * weights[1]
        loss_cross = self.edge_crossing_penalty(pos, edges) * weights[2]
        loss_bound = self.boundary_penalty(pos) * weights[3]
        return loss_stress + loss_overlap + loss_cross + loss_bound