import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_circle_positions(N, center=(400, 300), radius=200):
    angles = torch.linspace(0, 2*np.pi, N, dtype=torch.float)
    x = center[0] + radius * torch.cos(angles)
    y = center[1] + radius * torch.sin(angles)
    return torch.stack([x, y], dim=1)

def segments_intersect(p1, p2, q1, q2):
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    o1 = cross(p1, p2, q1)
    o2 = cross(p1, p2, q2)
    o3 = cross(q1, q2, p1)
    o4 = cross(q1, q2, p2)
    return o1*o2 < 0 and o3*o4 < 0

def count_edge_crossings(pos, edges):
    crossings = 0
    n = len(edges)
    pos_np = pos.detach().cpu().numpy()
    for i in range(n):
        for j in range(i+1, n):
            a1, a2 = edges[i]
            b1, b2 = edges[j]
            if len({a1, a2, b1, b2}) < 4:
                continue
            if segments_intersect(pos_np[a1], pos_np[a2], pos_np[b1], pos_np[b2]):
                crossings += 1
    return crossings

def plot_graph(graph, renderer=None, title=None, save_path=None):
    if title is None:
        title = graph.name
    pos = graph.pos.detach().cpu().numpy()
    edges = graph.get_edges()

    if renderer is not None:
        plt.figure(figsize=(14, 6))
        # Positions
        plt.subplot(1, 2, 1)
        plt.scatter(pos[:, 0], pos[:, 1], c='red', s=100, edgecolors='black', linewidths=2, zorder=5)
        for (i, j) in edges:
            plt.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 'b-', linewidth=2, alpha=0.8)
        plt.xlim(0, graph.width-1)
        plt.ylim(0, graph.height-1)
        plt.gca().invert_yaxis()
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
        plt.title(f"{title} - positions")
        plt.gca().set_aspect('equal')
        plt.grid(True, linestyle='--', alpha=0.3)

        # Rendu
        plt.subplot(1, 2, 2)
        img = renderer.render(graph.pos, edges).detach().cpu().numpy()
        if img.max() > 0:
            img = img / img.max()
        plt.imshow(img, cmap='gray', origin='upper', extent=[0, graph.width-1, graph.height-1, 0],
                   vmin=0, vmax=1)
        plt.title("Rendu (image)")
        plt.colorbar()
    else:
        plt.figure(figsize=(6, 6))
        plt.scatter(pos[:, 0], pos[:, 1], c='red', s=100, edgecolors='black', linewidths=2, zorder=5)
        for (i, j) in edges:
            plt.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 'b-', linewidth=2, alpha=0.8)
        plt.xlim(0, graph.width-1)
        plt.ylim(0, graph.height-1)
        plt.gca().invert_yaxis()
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
        plt.title(title)
        plt.gca().set_aspect('equal')
        plt.grid(True, linestyle='--', alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()