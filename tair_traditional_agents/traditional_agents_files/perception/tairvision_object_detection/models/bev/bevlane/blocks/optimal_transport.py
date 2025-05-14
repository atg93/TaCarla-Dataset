import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class OptimalTransport(nn.Module):
    def __init__(self, x_slide, y_slide):
        super(OptimalTransport, self).__init__()

        self.bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.position_encoder = None
        self.sinkhorn_iterations = 20

        self.x_slide = x_slide
        self.y_slide = y_slide

    def forward(self, node_features, active_nodes, node_positions, node_categories=None, node_order=None,
                match_points=None):
        similarity_matrices = []
        losses = []

        for batch_idx in range(len(node_features)):

            features = node_features[batch_idx][:, active_nodes[batch_idx]]
            if features.shape[1] == 0:
                similarity_matrices.append(None)
                continue
            scores = torch.einsum('dn,dm->nm', features, features)
            feat_dim = features.shape[0]
            scores = scores / feat_dim ** .5
            scores.fill_diagonal_(-float('inf'))
            scores = log_optimal_transport(scores[None, :], self.bin_score, self.sinkhorn_iterations)
            scores = scores.exp()

            if match_points is not None:
                node_cats = torch.ones_like(node_categories[batch_idx]) * -1
                node_ord = torch.zeros_like(node_order[batch_idx])
                if match_points[batch_idx] is not None:
                    node_cats[match_points[batch_idx][:, 0, 0], match_points[batch_idx][:, 0, 1]] = \
                        node_categories[batch_idx][match_points[batch_idx][:, 1, 0], match_points[batch_idx][:, 1, 1]]

                    node_ord[match_points[batch_idx][:, 0, 0], match_points[batch_idx][:, 0, 1]] = \
                        node_order[batch_idx][match_points[batch_idx][:, 1, 0], match_points[batch_idx][:, 1, 1]]

                adj_matrix = self.target_adjancency_matrix(node_cats, active_nodes[batch_idx],
                                                           node_positions[batch_idx], node_ord)
                loss = -torch.log(scores[0][adj_matrix == 1])
                losses.append(loss)
                similarity_matrices.append(scores)
            else:
                cost_matrix = self.spatial_adjacency_matrix(active_nodes[batch_idx], node_positions[batch_idx],
                                                            # only_up=True
                                                            )
                scores[0, :-1, :-1] = scores[:, :-1, :-1] * cost_matrix
                similarity_matrices.append(scores)

        if match_points is not None:
            if len(losses) == 0:
                return 0, similarity_matrices
            else:
                return torch.cat(losses).mean(), similarity_matrices
        else:
            return 0, similarity_matrices

    def spatial_adjacency_matrix(self, active_nodes, node_positions, only_up=False):
        positions = node_positions[active_nodes]
        y_a, y_b = torch.meshgrid((positions[..., 0], positions[..., 0]))
        y_dist_mat = (y_a - y_b).abs()
        y_dist_mat.fill_diagonal_(1000000)

        a, b = torch.meshgrid((positions[..., 1], positions[..., 1]))
        x_dist_mat = (a - b).abs()
        x_dist_mat.fill_diagonal_(1000000)

        cost_matrix = torch.logical_and(x_dist_mat <= self.x_slide * 2.0, y_dist_mat <= self.y_slide * 2.0)
        if only_up:
            cost_matrix = torch.logical_and(y_a < y_b, cost_matrix)
        return cost_matrix

    def target_adjancency_matrix(self, node_id_map, active_nodes, node_positions, node_order):
        labels_a, labels_b = torch.meshgrid(node_id_map[active_nodes], node_id_map[active_nodes])

        dist_mat = node_order[active_nodes][:, None] - node_order[active_nodes][None, :]
        dist_mat.fill_diagonal_(1e10)
        dist_mat[labels_a == -1] = 1e10
        dist_mat[labels_b == -1] = 1e10
        dist_mat[labels_a != labels_b] = 1e10
        dist_mat[dist_mat < 0] = 1e10

        adjacency_matrix = torch.zeros_like(labels_a, dtype=torch.bool)
        source, target = linear_sum_assignment(dist_mat.cpu().detach().numpy())
        mask = (dist_mat[source, target] < 1e9).cpu().numpy()
        adjacency_matrix[source[mask], target[mask]] = True
        adjacency_matrix[target[mask], source[mask]] = True
        adjacency_matrix.fill_diagonal_(False)

        adj_nobin_matrix = torch.zeros((active_nodes.sum() + 1, active_nodes.sum() + 1), device=active_nodes.device)
        adj_nobin_matrix[:-1, :-1] = adjacency_matrix
        adj_nobin_matrix[-1, :-1] = (adjacency_matrix == 0).all(dim=0)
        adj_nobin_matrix[:-1, -1] = (adjacency_matrix == 0).all(dim=1)
        adj_nobin_matrix[-1, -1] = 1

        return adj_nobin_matrix


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z
