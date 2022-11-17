data {
    int<lower=1> K;  // num components
    int<lower=1> D;  // embedding dimension
    int<lower=1> N;  // number of observations

    array[N] vector[D] X;  // data

    int<lower=1> N_out;  // number of held-out observations
    array[N_out] vector[D] X_out;  // held-out data
    
    // hyperparams
    vector[K] alpha;  // Dirichlet prior on theta
    array[K] vector[D] mu; // prior mean for beta
    array[K] real<lower=0.1> lambda; // prior variance for beta
    real<lower=0.1> gamma; // scale for half-Normal prior on sigma, the stdevs for beta_ij
    
}

parameters {
    simplex[K] theta;
    array[K] vector[D] beta;
    array[K] cholesky_factor_corr[D] L; // cholesky factor of covariance
    array[K] vector<lower=0.01>[D] sigma; // standard deviations of each beta_ij
}

transformed parameters {
    vector[K] log_theta = log(theta); // precompute reused value

    // compute the cholesky factor of the covariance matrix.
    // since COV = diag(sigma)LL^T(diag(sigma))^T, we have
    // that chol(COV) = diag(sigma)L
    array[K] cholesky_factor_cov[D] L_Sigma;

    for (k in 1:K) {
        L_Sigma[k] = diag_pre_multiply(sigma[k], L[k]);
    }
}

model {
    // draw theta from a Dirichlet
    theta ~ dirichlet(alpha);
    
    // draw each beta, L, and sigma
    for (k in 1:K) {
        beta[k] ~ normal(mu[k], lambda[k]);
        L[k] ~ lkj_corr_cholesky(1); // uniform matrix prior
        // draw sigma from Half-Normal(0, gamma)
        sigma[k] ~ normal(0, gamma);
    }
    
    // calculate contribution from each component
    for (n in 1:N) {
        // log_p_cluster[i] represents log p(X | z = i, beta[i]) where z is the cluster assignment
        vector[K] log_p_cluster;
        for (k in 1:K) {
            log_p_cluster[k] = multi_normal_cholesky_lpdf(X[n] | beta[k], L_Sigma[k]);
        }
        // marginalized log likelihood p(x | beta, theta)
        target += log_sum_exp(log_theta + log_p_cluster);
    }
}

generated quantities {
    // log prob of x_out under the current posterior draw
    real log_p_out = 0;
    
    for (n in 1:N_out) {
        vector[K] log_p_cluster;
        for (k in 1:K) {
            log_p_cluster[k] = multi_normal_cholesky_lpdf(X_out[n] | beta[k], L_Sigma[k]);
        }
        // marginalized log likelihood p(x | beta, theta)
        log_p_out += log_sum_exp(log_theta + log_p_cluster);
    }
}
