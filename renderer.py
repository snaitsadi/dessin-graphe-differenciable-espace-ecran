import torch

class DifferentiableRenderer:
    def __init__(self, height=600, width=800, sigma_node=4.0, sigma_edge=1.5, device='cpu'):
        self.H = height
        self.W = width
        self.sigma_node = sigma_node
        self.sigma_edge = sigma_edge
        self.device = device
        y = torch.arange(self.H, device=device).float()
        x = torch.arange(self.W, device=device).float()
        self.grid_y, self.grid_x = torch.meshgrid(y, x, indexing='ij')

    def render_nodes(self, pos):
        N = pos.shape[0]
        dx = self.grid_x.unsqueeze(0) - pos[:, 0, None, None]
        dy = self.grid_y.unsqueeze(0) - pos[:, 1, None, None]
        dist2 = dx**2 + dy**2
        node_img = torch.exp(-dist2 / (2 * self.sigma_node**2))
        return node_img.sum(dim=0)

    def render_edges(self, pos, edges):
        if len(edges) == 0:
            return torch.zeros(self.H, self.W, device=self.device)
        edge_img = torch.zeros(self.H, self.W, device=self.device)
        for (i, j) in edges:
            p1 = pos[i]
            p2 = pos[j]
            v = p2 - p1
            v_norm2 = torch.dot(v, v) + 1e-8
            w_x = self.grid_x - p1[0]
            w_y = self.grid_y - p1[1]
            t = (w_x * v[0] + w_y * v[1]) / v_norm2
            t = torch.clamp(t, 0, 1)
            proj_x = p1[0] + t * v[0]
            proj_y = p1[1] + t * v[1]
            dist2 = (self.grid_x - proj_x)**2 + (self.grid_y - proj_y)**2
            edge_seg = torch.exp(-dist2 / (2 * self.sigma_edge**2))
            edge_img += edge_seg
        return edge_img

    def render(self, pos, edges):
        return self.render_nodes(pos) + self.render_edges(pos, edges)