"""
Visualisation du graphe et de son rendu.
"""
import matplotlib.pyplot as plt

def plot_graph(graph, renderer=None, title="Graphe"):
    """Affiche le graphe (positions) et éventuellement l'image rendue."""
    pos = graph.pos.detach().numpy()
    edges = graph.get_edges()
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.scatter(pos[:,0], pos[:,1], c='red', s=100)
    for (i,j) in edges:
        plt.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], 'b-')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title(title)

    if renderer is not None:
        plt.subplot(1,2,2)
        img = renderer.render(graph.pos, edges).detach().numpy()
        plt.imshow(img, cmap='gray', origin='lower', extent=[0,1,0,1])
        plt.title("Rendu")
    plt.show()