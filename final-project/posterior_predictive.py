import torch
from torch.distributions import Beta, Gamma, Poisson, Categorical
from torch_mixture_class import MixtureSameFamily


def mix_weights(beta):
    weights = torch.zeros(beta.shape[0] + 1)
    for t in range(beta.shape[0]):
        weights[t] = beta[t] * torch.prod(1. - beta[:t], dim=0)
    weights[beta.shape[0]] = 1. - torch.sum(weights)
    normalized_weights = weights.clone()
    return weights / normalized_weights


def log_posterior_predictive_eval(x_new, kappa, tau_0, tau_1, S):
    T = kappa.shape[0] + 1
    q_beta = Beta(torch.ones(T - 1), kappa)
    q_lambda = Gamma(tau_0, tau_1)
    beta_mc = q_beta.sample([S])
    lambda_mc = q_lambda.sample([S])
    log_prob = 0
    for s in range(S):
        post_pred_weights = mix_weights(beta_mc[s])
        post_pred_clusters = lambda_mc[s]
        for t in range(post_pred_clusters.shape[0]):
            log_prob -= post_pred_weights[t] * torch.exp(
                Poisson(post_pred_clusters[t]).log_prob(x_new))
    log_prob /= S
    return log_prob


def sq_log_posterior_predictive_eval(x_new, kappa, tau_0, tau_1, S):
    T = kappa.shape[0] + 1
    q_beta = Beta(torch.ones(T - 1), kappa)
    q_lambda = Gamma(tau_0, tau_1)
    beta_mc = q_beta.sample([S])
    lambda_mc = q_lambda.sample([S])
    log_prob = 0
    for s in range(S):
        post_pred_weights = mix_weights(beta_mc[s])
        post_pred_clusters = lambda_mc[s]
        for t in range(post_pred_clusters.shape[0]):
            log_prob -= post_pred_weights[t] * torch.exp(
                Poisson(
                    post_pred_clusters[t]).log_prob(x_new)) ** 2
    log_prob /= S
    return log_prob


def posterior_predictive_sample(kappa, tau_0, tau_1, S, M):
    T = kappa.shape[0] + 1
    q_beta = Beta(torch.ones(T - 1), kappa)
    q_lambda = Gamma(tau_0, tau_1)
    beta_mc = q_beta.sample([S])
    lambda_mc = q_lambda.sample([S])

    hallucinated_samples = torch.zeros(S, M)
    for s in range(S):
        post_pred_weights = mix_weights(beta_mc[s])
        post_pred_clusters = lambda_mc[s]
        hallucinated_samples[s, :] = MixtureSameFamily(Categorical(
            post_pred_weights), Poisson(post_pred_clusters)).sample([M])

    return hallucinated_samples
