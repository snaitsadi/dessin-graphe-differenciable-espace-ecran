import torch
import torch.optim as optim

def optimize_graph(graph, renderer, losses, n_iter=1000, lr=5.0, weights=(1.0, 0.5, 3.0, 0.1),
                   verbose=True, return_history=False):
    pos = graph.pos.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([pos], lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    edges = graph.get_edges()
    adj = graph.adj

    history = {'pos': [], 'loss': []} if return_history else None

    for i in range(n_iter):
        optimizer.zero_grad()
        loss = losses.total_loss(pos, adj, edges, weights=weights)
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            pos[:, 0].clamp_(0, graph.width-1)
            pos[:, 1].clamp_(0, graph.height-1)

        if return_history:
            history['pos'].append(pos.clone().detach().cpu())
            history['loss'].append(loss.item())

        if verbose and i % 200 == 0:
            print(f"Iter {i}, loss = {loss.item():.4f}")

    if return_history:
        return pos.detach(), history
    else:
        return pos.detach()