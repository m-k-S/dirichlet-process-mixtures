import numpy as np
from scipy.special import betaln, digamma, gammaln, logsumexp
from sklearn.cluster import KMeans

def estimate_parameters(assignments, data):
    assignment_counts = assignments.sum(axis=0) + 1e-15
    print (assignment_counts.shape)
    means = np.dot(assignments.T, data) / assignment_counts[:, np.newaxis]
    sq_means = 2 * means * np.dot(assignments.T, data) / assignment_counts[:, np.newaxis]
    covariances = np.dot(assignments.T, data * data) / assignment_counts[:, np.newaxis] - sq_means + means**2 + 0.1
    return assignment_counts, means, covariances

def estimate_weights(assignment_counts):
    weight_concentration = (1. + assignment_counts, ((1. / T) + np.hstack((np.cumsum(assignment_counts[::-1])[-2::-1], 0))))
    return weight_concentration

def estimate_means(data, assignment_counts, means):
    new_mean_prec = 1. + assignment_counts
    new_means = ((1. * data.mean(axis=0)[:, np.newaxis] + assignment_counts[:, np.newaxis] * means) / new_mean_prec[:, np.newaxis])
    return new_means, new_mean_prec

def estimate_precisions(assignment_counts, means, covariances):
    new_dof = T + assignment_counts
    diff = means - data.mean(axis=0)[:, np.newaxis]
    new_covariances = (D + assignment_counts[:, np.newaxis] * (covariances + (1. / mean_prec)[:, np.newaxis] * np.square(diff)))
    new_covariances /= new_dof[:, np.newaxis]

    new_precisions = 1. / np.sqrt(new_covariances)
    return new_covariances, new_precisions, new_dof

def initialization(data):
    global mean_prec
    global dof

    label_matrix = np.zeros((data.shape[0], T))
    assignments = KMeans(n_clusters=T, n_init=1).fit(data).labels_
    label_matrix[np.arange(data.shape[0]), assignments] = 1

    counts, means, covariances = estimate_parameters(data, label_matrix)
    weight_conc = estimate_weights(counts)
    new_means, mean_prec = estimate_means(data, counts, means)
    new_covariances, new_precisions, dof = estimate_precisions(counts, means, covariances)
    return weight_conc, new_means, new_covariances, new_precisions

def compute_expectations(data, means, precisions_pd, concentrations):
    log_determinant = lambda x: np.sum(np.log(x), axis=1)

    print (log_determinant(precisions_pd).shape)
    # Estimate log Gaussian probability (requires means, Cholesky precision matrix, data, dof)
    precisions = precisions_pd ** 2
    log_prob = (np.sum((means ** 2 * precisions), 1) - 2. * np.dot(data, (means * precisions).T) + np.dot(data ** 2, precisions).T)
    log_prob_g = (-.5 * (D * np.log(2 * np.pi) + log_prob) + log_determinant(precisions_pd)) - (.5 * D * np.log(dof))

    # Estimate total log probability (requires data, dof)
    log_prob_l = D * np.log(2.) + np.sum(digamma(.5 * (dof - np.arange(0, D)[:, np.newaxis])), 0)
    log_prob_total = log_prob_g + 0.5 * (log_prob_l - D / data.mean(axis=0)[:, np.newaxis])

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
    new_covariances, new_precisions, dof = estimate_precisions(counts, means, covariances)
    return weight_conc, new_means, new_covariances, new_precisions

def elbo(log_likelihoods, mean_lwp, precisions, weight_concentrations):
    log_determinant = (np.sum(np.log(precisions), axis=1) - 0.5 * D * np.log(dof))

    log_norm = -(dof * log_determinant +
                dof * D * .5 * np.log(2.) +
                np.sum(gammaln(.5 * (dof - np.arange(D)[:, np.newaxis])), 0))

    log_norm = np.sum(log_norm)
    log_norm_weight = -np.sum(betaln(weight_concentrations[0], weight_concentrations[1]))
    return (-np.sum(np.exp(log_likelihoods) * log_likelihoods) - log_norm - log_norm_weight - 0.5 * D * np.sum(np.log(mean_prec)))

T = 15
N, D = data.shape

mean_prec = 1.
dof = T

def dpmm(data, n_iter):
    concentrations, means, covariances, precisions = initialization(data)
    lower_bound = -np.infty

    for i in tqdm(range(1, n_iter+1)):
        prev_low_bound = lower_bound
        mean_lwp, log_likelihoods = compute_expectations(data, means, precisions, concentrations)
        concentrations, means, covariances, precisions = maximize_probabilities(data, log_likelihoods)
        lower_bound = elbo(log_likelihoods, mean_lwp, precisions, concentrations)

    _, log_likelihoods = compute_expectations(data, means, precisions, concentrations)

    return log_likelihoods.argmax(axis=1)


dpmm_labels = dpmm(data, 1000)
