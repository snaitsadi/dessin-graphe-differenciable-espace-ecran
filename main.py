import torch
from graph import Graph
from renderer import DifferentiableRenderer
from losses import GraphLayoutLosses
from optimization import optimize_graph
from visualization import plot_graph, generate_circle_positions, count_edge_crossings

def main():
    print("="*60)
    print("Dessin de graphe différentiable - Version optimisée")
    print("Avec forte pénalité de croisement et rendu net")
    print("="*60)

    # Paramètres
    W, H = 800, 600
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(f"Dimensions: {W}x{H} pixels")
    print(f"Device: {device}")

    # Renderers
    renderer_opt = DifferentiableRenderer(height=H, width=W, sigma_node=4.0, sigma_edge=1.5, device=device)
    renderer_disp = DifferentiableRenderer(height=H, width=W, sigma_node=2.0, sigma_edge=1.0, device=device)

    losses = GraphLayoutLosses(renderer_opt, scale=50.0)

    # --------------------------------------------------------
    # Exemple 1 : Cycle (pas de croisements à l'optimum)
    # --------------------------------------------------------
    print("\n--- Exemple 1: Cycle avec initialisation aléatoire ---")
    N = 12
    adj = torch.zeros(N, N)
    for i in range(N):
        adj[i, (i+1)%N] = 1
        adj[(i+1)%N, i] = 1
    # Initialisation aléatoire (pour voir si l'optimisation trouve le cercle)
    g_cycle = Graph(adj, name="Cycle", width=W, height=H)
    g_cycle.to(device)

    print("Avant optimisation:")
    plot_graph(g_cycle, renderer_disp, title="Cycle - initial")

    # Optimisation avec poids renforcés sur crossing
    opt_pos, history = optimize_graph(g_cycle, renderer_opt, losses, n_iter=1000, lr=5.0,
                                       weights=(1.0, 0.5, 3.0, 0.1), return_history=True)
    g_cycle_opt = Graph(adj, opt_pos, name="Cycle optimisé", width=W, height=H)

    print("Après optimisation:")
    plot_graph(g_cycle_opt, renderer_disp, title="Cycle - après optimisation")

    # Évaluation des croisements
    crossings_before = count_edge_crossings(g_cycle.pos, g_cycle.get_edges())
    crossings_after = count_edge_crossings(g_cycle_opt.pos, g_cycle_opt.get_edges())
    print(f"Nombre de croisements avant: {crossings_before}, après: {crossings_after}")

    # Courbe de loss
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(history['loss'])
    plt.xlabel("Itération")
    plt.ylabel("Loss")
    plt.title("Évolution de la loss (cycle)")
    plt.grid()
    plt.show()

    # --------------------------------------------------------
    # Exemple 2 : Petit graphe aléatoire
    # --------------------------------------------------------
    print("\n--- Exemple 2: Petit graphe aléatoire ---")
    N2 = 8
    adj_rand = torch.rand(N2, N2) < 0.4
    adj_rand = torch.triu(adj_rand, 1)
    adj_rand = adj_rand + adj_rand.T
    adj_rand = adj_rand.float()
    g_rand = Graph(adj_rand, name="Aléatoire", width=W, height=H)
    g_rand.to(device)

    print("Avant optimisation:")
    plot_graph(g_rand, renderer_disp, title="Aléatoire - initial")

    opt_pos_rand, history_rand = optimize_graph(g_rand, renderer_opt, losses, n_iter=800, lr=5.0,
                                                 weights=(1.0, 0.5, 3.0, 0.1), return_history=True)
    g_rand_opt = Graph(adj_rand, opt_pos_rand, name="Aléatoire optimisé", width=W, height=H)

    print("Après optimisation:")
    plot_graph(g_rand_opt, renderer_disp, title="Aléatoire - après optimisation")

    crossings_before_rand = count_edge_crossings(g_rand.pos, g_rand.get_edges())
    crossings_after_rand = count_edge_crossings(g_rand_opt.pos, g_rand_opt.get_edges())
    print(f"Nombre de croisements avant: {crossings_before_rand}, après: {crossings_after_rand}")

    print("\nTerminé.")

if __name__ == "__main__":
    main()