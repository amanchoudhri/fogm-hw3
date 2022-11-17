data {
    int<lower=1> K;  // num components
    int<lower=1> D;  // embedding dimension
    int<lower=1> N;  // number of observations

    array[N] vector[D] X;  // data
    
    // hyperparams
    vector[K] alpha;  // Dirichlet prior on theta
    array[K] vector[D] mu; // prior mean for beta
    array[K] real<lower=0.1> lambda; // prior variance for beta
    real<lower=0.1> gamma; // scale for half-Normal prior on sigma, the stdevs for beta_ij
    
    int<lower=1> N_out;  // number of held-out observations

    array[N_out] vector[D] X_out;  // held-out data
}

parameters {
    simplex[K] theta;
    array[K] vector[D] beta;

    array[K] vector<lower=0.01>[D] sigma; // standard deviations of each beta_ij
}

transformed parameters {
    // precompute reused value
    vector[K] log_theta = log(theta);
}
model {
    // draw theta from a Dirichlet
    theta ~ dirichlet(alpha);
    
    // draw parameters for each component
    for (k in 1:K) {
        beta[k] ~ normal(mu[k], lambda[k]);
        // draw each sigma from a half-Normal (the half comes since we bounded
        // sigma from below at 0)
        sigma[k] ~ normal(0, gamma);
    }

    // calculate contribution from each component
    for (n in 1:N) {
        // log_p_cluster[i] represents log p(x | z = i, beta[i]) where z is the cluster assignment
        vector[K] log_p_cluster;
        for (k in 1:K) {
            log_p_cluster[k] = normal_lpdf(X[n] | beta[k], sigma[k]);
        }
        // marginalized log likelihood p(x | beta, theta)
        target += log_sum_exp(log_theta + log_p_cluster);
    }
}

generated quantities {
    // draw from posterior predictive
    // int z_tilde = categorical_rng(theta);
    // array[D] real x_tilde = normal_rng(beta[z_tilde], sigma[z_tilde]);
    
    // log prob of x_out under the current posterior draw
    real log_p_out = 0;
    
    for (n in 1:N_out) {
        vector[K] log_p_cluster;
        for (k in 1:K) {
            log_p_cluster[k] = normal_lpdf(X_out[n] | beta[k], sigma[k]);
        }
        // marginalized log likelihood p(x | beta, theta)
        log_p_out += log_sum_exp(log_theta + log_p_cluster);
    }
}
