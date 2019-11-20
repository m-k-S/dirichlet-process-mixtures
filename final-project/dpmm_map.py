import numpy as np
from scipy.special import betaln, digamma, gammaln, logsumexp
from sklearn.cluster import KMeans

# Hyperparameters
N, D = None, None
mean_prec = 1.
dof = 64.
alpha = 0.5
gamma = 1.5
T = 16
truncation = 0.02

def estimate_parameters(data, assignments):
    assignment_counts = assignments.sum(axis=0) + 1e-15
    means = np.dot(assignments.T, data) / assignment_counts[:, np.newaxis]
    sq_means = 2 * means * np.dot(assignments.T, data) / assignment_counts[:, np.newaxis]
    covariances = np.dot(assignments.T, data * data) / assignment_counts[:, np.newaxis] - sq_means + means**2 + 1e-7
    return assignment_counts, means, covariances

def estimate_weights(assignment_counts):
    weight_concentration = (1. + assignment_counts, (alpha + np.hstack((np.cumsum(assignment_counts[::-1])[-2::-1], 0))))
    return weight_concentration

def estimate_means(data, assignment_counts, means):
    new_mean_prec = 1. + assignment_counts
    new_means = ((1. * data.mean(axis=0) + assignment_counts[:, np.newaxis] * means) / new_mean_prec[:, np.newaxis])
    return new_means, new_mean_prec

def estimate_precisions(data, assignment_counts, means, covariances):
    new_dof = D + assignment_counts
    diff = means - data.mean(axis=0)
    new_covariances = (gamma * np.var(data, axis=0, ddof=1) + assignment_counts[:, np.newaxis] * (covariances + (1. / mean_prec)[:, np.newaxis] * np.square(diff)))
    new_covariances /= new_dof[:, np.newaxis]

    new_precisions = 1. / np.sqrt(new_covariances)
    return new_covariances, new_precisions, new_dof

def initialization(data):
    global mean_prec
    global dof

    label_matrix = np.zeros((N, T))
    assignments = KMeans(n_clusters=T, n_init=1).fit(data).labels_
    label_matrix[np.arange(N), assignments] = 1

    counts, means, covariances = estimate_parameters(data, label_matrix)
    weight_conc = estimate_weights(counts)
    new_means, mean_prec = estimate_means(data, counts, means)
    new_covariances, new_precisions, dof = estimate_precisions(data, counts, means, covariances)

    return weight_conc, new_means, new_covariances, new_precisions

def compute_expectations(data, means, precisions_pd, concentrations):
    log_determinant = lambda x: np.sum(np.log(x), axis=1)
    # Estimate log Gaussian probability (requires means, Cholesky precision matrix, data, dof)
    precisions = precisions_pd ** 2
    log_prob = (np.sum((means ** 2 * precisions), 1) - 2. * np.dot(data, (means * precisions).T) + np.dot(data ** 2, precisions.T))
    log_prob_g = (-.5 * (D * np.log(2 * np.pi) + log_prob) + log_determinant(precisions_pd)) - (.5 * D * np.log(dof))

    # Estimate total log probability (requires data, dof)
    log_prob_l = D * np.log(2.) + np.sum(digamma(.5 * (dof - np.arange(0, D)[:, np.newaxis])), 0)
    log_prob_total = log_prob_g + 0.5 * (log_prob_l - D / mean_prec)

    # Estimate log weights
    log_weights = digamma(concentrations[0]) - digamma(concentrations[0] + concentrations[1]) \
                    + np.hstack((0, np.cumsum(digamma(concentrations[1]) - digamma(concentrations[0] + concentrations[1]))[:-1]))

    # Estimate log likelihoods
    log_weighted_prob = log_weights + log_prob_total
    softmax_log_weighted_prob = logsumexp(log_weighted_prob, axis=1)
    with np.errstate(under='ignore'):
        log_likelihoods = log_weighted_prob - softmax_log_weighted_prob[:, np.newaxis]
    mean_log_weighted_prob = np.mean(softmax_log_weighted_prob)

    return mean_log_weighted_prob, log_likelihoods # shapes are also NxD

def maximize_probabilities(data, log_likelihoods):
    counts, means, covariances = estimate_parameters(data, np.exp(log_likelihoods))
    weight_conc = estimate_weights(counts)
    new_means, mean_prec = estimate_means(data, counts, means)
    new_covariances, new_precisions, dof = estimate_precisions(data, counts, means, covariances)
    return weight_conc, new_means, new_covariances, new_precisions

def final_truncation(data, means, precisions_pd, concentrations, final_dof, final_mean_p):
    log_determinant = lambda x: np.sum(np.log(x), axis=1)

    # Estimate log Gaussian probability (requires means, Cholesky precision matrix, data, dof)
    precisions = precisions_pd ** 2
    log_prob = (np.sum((means ** 2 * precisions), 1) - 2. * np.dot(data, (means * precisions).T) + np.dot(data ** 2, precisions.T))
    log_prob_g = (-.5 * (D * np.log(2 * np.pi) + log_prob) + log_determinant(precisions_pd)) - (.5 * D * np.log(final_dof))

    # Estimate total log probability (requires data, dof)
    log_prob_l = D * np.log(2.) + np.sum(digamma(.5 * (final_dof - np.arange(0, D)[:, np.newaxis])), 0)
    log_prob_total = log_prob_g + 0.5 * (log_prob_l - D / final_mean_p)

    # Estimate log weights
    log_weights = digamma(concentrations[0]) - digamma(concentrations[0] + concentrations[1]) \
                    + np.hstack((0, np.cumsum(digamma(concentrations[1]) - digamma(concentrations[0] + concentrations[1]))[:-1]))

    # Estimate log likelihoods
    log_weighted_prob = log_weights + log_prob_total
    softmax_log_weighted_prob = logsumexp(log_weighted_prob, axis=1)
    with np.errstate(under='ignore'):
        log_likelihoods = log_weighted_prob - softmax_log_weighted_prob[:, np.newaxis]

    return log_likelihoods

def elbo(log_likelihoods, mean_lwp, precisions, weight_concentrations):
    log_determinant = (np.sum(np.log(precisions), axis=1) - 0.5 * D * np.log(dof))

    log_norm = -(dof * log_determinant +
                dof * D * .5 * np.log(2.) +
                np.sum(gammaln(.5 * (dof - np.arange(D)[:, np.newaxis])), 0))

    log_norm = np.sum(log_norm)
    log_norm_weight = -np.sum(betaln(weight_concentrations[0], weight_concentrations[1]))
    return (-np.sum(np.exp(log_likelihoods) * log_likelihoods) - log_norm - log_norm_weight - 0.5 * D * np.sum(np.log(mean_prec)))

def dpmm(data, hyperparameters):
    global N
    global D
    global mean_prec
    global dof
    global alpha
    global gamma
    global T
    global truncation

    N, D = data.shape
    n_iter = hyperparameters['n_iter']
    mean_prec = hyperparameters['mean_prec']
    dof = hyperparameters['dof']
    alpha = hyperparameters['alpha']
    gamma = hyperparameters['gamma']
    T = hyperparameters['T']
    truncation = hyperparameters['truncation']

    concentrations, means, covariances, precisions = initialization(data)
    elbos = []

    for i in range(1, n_iter+1):
        mean_lwp, log_likelihoods = compute_expectations(data, means, precisions, concentrations)
        concentrations, means, covariances, precisions = maximize_probabilities(data, log_likelihoods)
        elbos.append(elbo(log_likelihoods, mean_lwp, precisions, concentrations))

    final_weights = np.exp(digamma(concentrations[0]) - digamma(concentrations[0] + concentrations[1]) \
                    + np.hstack((0, np.cumsum(digamma(concentrations[1]) - digamma(concentrations[0] + concentrations[1]))[:-1])))

    final_means = means[final_weights > truncation]
    final_precisions = precisions[final_weights > truncation]
    final_concentrations = (concentrations[0][final_weights > truncation], concentrations[1][final_weights > truncation])
    final_dof = dof[final_weights > truncation]
    final_mean_prec = mean_prec[final_weights > truncation]

    log_likelihoods = final_truncation(data, final_means, final_precisions, final_concentrations, final_dof, final_mean_prec)

    return log_likelihoods.argmax(axis=1), elbos
