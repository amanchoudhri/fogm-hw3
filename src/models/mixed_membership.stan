data {
    int<lower=1> K;  // num components
    int<lower=1> D;  // embedding dimension
    int<lower=1> F;  // number of families
    int<lower=1> N;  // number of observations
    array[F] int<lower=1> lens;  // number of observations per family

    // data; assume that each family's data is concatenated together
    array[N] vector[D] X;
    
    // hyperparams
    vector[K] alpha;  // Dirichlet prior on theta
    array[K] vector[D] mu; // prior mean for beta
    array[K] real<lower=0.1> lambda; // prior variance for beta
    real<lower=0.1> gamma; // scale for half-Normal prior on sigma, the stdevs for beta_ij
    
    int<lower=1> N_out;  // number of held-out observations

    array[N_out] vector[D] X_out;  // held-out data
    array[F] int<lower=1> lens_out;  // number of held-out observations per family
}

parameters {
    array[F] simplex[K] theta; // mixture proportions per family
    array[K] vector[D] beta;   // mixture components

    array[K] cholesky_factor_corr[D] L; // cholesky factor of covariance
    array[K] vector<lower=0.01>[D] sigma; // standard deviations of each beta_ij
}

transformed parameters {
    // precompute reused value
    array[F] vector[K] log_theta = log(theta);

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
    
    // draw parameters for each component
    for (k in 1:K) {
        beta[k] ~ normal(mu[k], lambda[k]);
        L[k] ~ lkj_corr_cholesky(1); // uniform correlation matrix prior
        // draw each sigma from a half-Normal (the half comes since we bounded
        // sigma from below at 0)
        sigma[k] ~ normal(0, gamma);
    }

    // draw the observations from each family

    int pos = 1;  // left endpoint of current family segment in the array
    // declare reused variable out here to save memory
    vector[K] log_p_cluster;

    for (i in 1:F) {
        // create the mixture log likelihood for each family
        for (j in 1:lens[i]) {
            for (k in 1:K) {
                // log_p_cluster[i] represents log p(x | z = i, beta[i]),
                // where z is the cluster assignment
                log_p_cluster[k] = multi_normal_cholesky_lpdf(X[pos + j - 1] | beta[k], L_Sigma[k]);
            }
            // marginalized log likelihood p(x | beta, theta)
            target += log_sum_exp(log_theta[i] + log_p_cluster);
        }
        pos += lens[i];
    }
}

generated quantities {
    // log prob of x_out under the current posterior draw
    real log_p_out = 0;
    
    int pos_out = 1;
    for (i in 1:F) {
        for (j in 1:lens_out[i]) {
            vector[K] log_p_cluster;
            for (k in 1:K) {
                // log_p_cluster[i] represents log p(x | z = i, beta[i]),
                // where z is the cluster assignment
                log_p_cluster[k] = multi_normal_cholesky_lpdf(X[pos_out + j - 1] | beta[k], L_Sigma[k]);
            }
            // marginalized log likelihood p(x | beta, theta)
            log_p_out += log_sum_exp(log_theta[i] + log_p_cluster);
        }
        pos_out += lens_out[i];
    }
}
