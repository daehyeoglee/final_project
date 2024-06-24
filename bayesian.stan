data {
  int<lower=1> N;
  int<lower=0> mi[N];
  int<lower=0> ni[N];
  real td[N];
  real dd1[N];
  real dd2[N];
}

parameters {
  real<lower=0, upper=1> param; // leaking factor k
}

model {
  vector[N] w1;
  vector[N] w2;
  vector[N] t_hat1;
  vector[N] t_hat2;
  vector[N] t_diff;
  vector[N] phi;
  
  // Prior distribution
  param ~ uniform(0, 1);
  
  // Likelihood function
  for (i in 1:N) {
    w1[i] = param^fabs(dd1[i] - td[i]);
    w2[i] = param^fabs(dd2[i] - td[i]);
    t_hat1[i] = (td[i] + dd1[i] * w1[i]) / (1 + w1[i]);
    t_hat2[i] = (td[i] + dd2[i] * w2[i]) / (1 + w2[i]);
    t_diff[i] = t_hat2[i] - t_hat1[i];
    phi[i] = normal_cdf(t_diff[i], 0, 1);
    target += mi[i] * log(phi[i]) + (ni[i] - mi[i]) * log(1 - phi[i]);
  }
}
