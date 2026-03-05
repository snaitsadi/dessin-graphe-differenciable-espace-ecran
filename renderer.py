"""
Rendu différentiable d'un graphe en image (grille de pixels).
"""
import torch

class DifferentiableRenderer:
      """Rendu d'un graphe sur une grille de pixels de façon différentiable."""
    def __init__(self, height=64, width=64, sigma_node=0.05, sigma_edge=0.02):
        self.H = height
        self.W = width
        self.sigma_node = sigma_node
        self.sigma_edge = sigma_edge
        # Grille de pixels (coordonnées normalisées entre 0 et 1)
        y = torch.linspace(0, 1, self.H)
        x = torch.linspace(0, 1, self.W)
        self.grid_y, self.grid_x = torch.meshgrid(y, x, indexing='ij')
        # Forme (H, W) pour les coordonnées


    def render_nodes(self, pos):
        """
        Rend les nœuds comme des taches gaussiennes.
        pos: (N, 2) positions dans [0,1]
        Retourne image (H, W)
        """
        dist2 = ( (self.grid_x[None, :, :] - pos[:, 0, None, None])**2 +
                  (self.grid_y[None, :, :] - pos[:, 1, None, None])**2 )
        node_img = torch.exp(-dist2 / (2 * self.sigma_node**2))
        return node_img.sum(dim=0)


    def render_edges(self, pos, edges):
        """
        Rend les arêtes comme des bandes gaussiennes.
        pos: (N, 2)
        edges: liste de tuples (i, j)
        Retourne image (H, W)
        """
        if len(edges) == 0:
            return torch.zeros(self.H, self.W)
        edge_img = torch.zeros(self.H, self.W)
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
        """Rendu combiné nœuds + arêtes."""
        node_img = self.render_nodes(pos)
        edge_img = self.render_edges(pos, edges)
        return node_img + edge_img