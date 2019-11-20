import numpy as np
import torch
from torch.distributions import *
from torch.autograd import Variable

def mix_weights(beta):
    weights = [beta[t] * torch.prod(1. - beta[:t], dim=0) for t in range(beta.shape[0])]
    weights += [1. - sum(weights)]
    return weights

def construct_priors(alpha, lambda_0, lambda_1, T):
    p_beta = Beta(1, alpha)
    p_lambda = Gamma(lambda_0, lambda_1)
    p_zeta = Categorical(torch.tensor(mix_weights(p_beta.rsample([T-1]))))
    return p_beta, p_lambda, p_zeta

def construct_variational_parameters(T, N):
    kappa = Variable(Uniform(0, 2).rsample([T-1]), requires_grad=True) # T-1 params for the second part of variational Beta factors
    tau_0 = Uniform(0, 100).rsample([T]) # T scale params for the variational Gamma factors
    tau_1 = LogNormal(0, 1).rsample([T]) # T rate params for the variational Gamma factors
    tau = Variable(torch.stack((tau_0, tau_1)).T, requires_grad = True)
    phi = Variable(Dirichlet(1/T * torch.ones(T)).rsample([N]), requires_grad=True) # N,T params for the variational Categorical factors

    return kappa, tau, phi

def construct_variational_family(kappa, tau, phi, T):
    q_beta = Beta(torch.ones(T-1), kappa)
    q_lambda = Gamma(tau[:, 0], tau[:, 1])
    q_zeta = Categorical(phi)
    return q_beta, q_lambda, q_zeta

def sample_monte_carlo(var_family, num_samples, N):
    q_beta, q_lambda, q_zeta = var_family
    z_mc = q_zeta.sample([num_samples])
    lambda_mc = q_lambda.sample([num_samples])
    rates_mc = torch.zeros(num_samples, N)
    for s in range(num_samples):
      for n in range(N):
        rates_mc[s, n] = lambda_mc[s, z_mc[s, n]]
    return z_mc, lambda_mc, rates_mc

def compute_lower_bound(priors, var_family, mc_samples, X, T, N):
    p_beta, p_lambda, p_zeta = priors
    q_beta, q_lambda, q_zeta = var_family
    pq_pairs = [[q_beta, p_beta], [q_lambda, p_lambda], [q_zeta, p_zeta]]
    kl_qp = sum([sum(kl_divergence(q, p)) for q, p in pq_pairs])

    z_mc, lambda_mc, rates_mc = mc_samples
    num_samples = lambda_mc.shape[0]
    px_mc = [Poisson(rates_mc[:, n]) for n in range(N)]
    log_probs = torch.zeros(num_samples, N)
    for n in range(N):
      log_probs[:, n] = px_mc[n].log_prob(X[n])

    mean_log_prob = torch.mean(log_probs, dim=0)
    sum_mean_log_prob = torch.sum(mean_log_prob)
    elbo = kl_qp - sum_mean_log_prob
    return elbo

def bbvi(data, hyperparameters):
    alpha = hyperparameters['alpha']
    lambda_0 = hyperparameters['lambda_0']
    lambda_1 = hyperparameters['lambda_1']
    T = hyperparameters['T']
    n_iter = hyperparameters['n_iter']
    num_samples = hyperparameters['num_samples']
    lr = hyperparameters['lr']

    N = data.shape[0]

    priors = construct_priors(alpha, lambda_0, lambda_1, T)
    kappa, tau, phi = construct_variational_parameters(T, N)
    var_family = construct_variational_family(kappa, tau, phi, T)
    mc_samples = sample_monte_carlo(var_family, num_samples, N)

    elbo = compute_lower_bound(priors, var_family, mc_samples, data, T, N)
    optimizer = torch.optim.Adam([kappa, tau, phi], lr=lr)

    for i in range(n_iter):
        optimizer.zero_grad()
        loss = elbo
        loss.backward(retain_graph=True)
        optimizer.step()
        with torch.no_grad():
          kappa = kappa.clamp(0, np.inf)
          tau = tau.clamp(0, np.inf)
          phi = phi / torch.sum(phi, dim=1).view(N, 1)

    return kappa, tau, phi
