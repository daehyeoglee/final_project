data {
  int<lower=1> N;
  int<lower=0> mi[N];
  int<lower=0> ni[N];
  real td[N];
  real dd1[N];
  real dd2[N];
  int<lower=1> J;
  int<lower=1,upper=J> participant[N];
}

parameters {
  real<lower=0, upper=1> mu_param;
  real<lower=0> sigma_param;
  real<lower=0, upper=1> param[J];
}

model {
  vector[N] w1;
  vector[N] w2;
  vector[N] t_hat1;
  vector[N] t_hat2;
  vector[N] t_diff;
  vector[N] phi;

  // Hyperparameters
  mu_param ~ uniform(0, 1);
  sigma_param ~ normal(0, 0.5);

  // Priors
  for (j in 1:J)
    param[j] ~ normal(mu_param, sigma_param);

  // Likelihood
  for (i in 1:N) {
    w1[i] = param[participant[i]]^fabs(dd1[i] - td[i]);
    w2[i] = param[participant[i]]^fabs(dd2[i] - td[i]);
    t_hat1[i] = (td[i] + dd1[i] * w1[i]) / (1 + w1[i]);
    t_hat2[i] = (td[i] + dd2[i] * w2[i]) / (1 + w2[i]);
    t_diff[i] = t_hat2[i] - t_hat1[i];
    phi[i] = normal_cdf(t_diff[i], 0, 1);
    target += mi[i] * log(phi[i]) + (ni[i] - mi[i]) * log(1 - phi[i]);
  }
}
