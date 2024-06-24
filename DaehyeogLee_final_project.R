#####
rm(list = ls())
set.seed(100)
library(dplyr)
library(rstan)
library(reshape2)
library(ggplot2)
#####

################################################
############### MLE model ######################
################################################
dclmData = read.csv('data_distractors.csv')
head(dclmData)

mle_dclm <- function(param, td, dd1, dd2, mi, ni) {
  w1 <- param^(abs(dd1 - td))
  w2 <- param^(abs(dd2 - td))
  t_hat1 <- (td + dd1*w1)/(1 + w1)
  t_hat2 <- (td + dd2*w2)/(1 + w2)
  t_diff <- t_hat2 - t_hat1
  phi <- pnorm(t_diff, mean = 0, sd = 1)
  
  loglik <- mi*log(phi) + (ni - mi)*log(1 - phi)
  -sum(loglik)
}

results <- data.frame(participant = integer(),
                      param_estimate = numeric(),
                      log_likelihood = numeric(),
                      convergence = integer(),
                      stringsAsFactors = FALSE)

subj <- dclmData %>%
  filter(participant == 15, condition == "distractors") %>%
  group_by(
    target_duration, distractors_duration_1, distractors_duration_2
  ) %>%
  summarize(
    count_ones = sum(response == 1),
    ni = n()) %>%
  ungroup() %>%
  rename(mi = count_ones)

for (p in 1:22) {
  subj <- dclmData %>%
    filter(participant == p, condition == "distractors") %>%
    group_by(
      target_duration, distractors_duration_1, distractors_duration_2
    ) %>%
    summarize(
      count_ones = sum(response == 1),
      ni = n()) %>%
    ungroup() %>%
    rename(mi = count_ones)
  
  td <- subj$target_duration
  dd1 <- subj$distractors_duration_1
  dd2 <- subj$distractors_duration_2
  mi <- subj$mi
  ni <- subj$ni
  
  param_init <- runif(1)
  param_low <- c(0)
  param_up <- c(1)
  
  dclm <- optim(
    par = param_init,
    fn = mle_dclm,
    gr = NULL,
    td = td, dd1 = dd1, dd2 = dd2, mi = mi, ni = ni,
    method = "L-BFGS-B",
    lower = param_low,
    upper = param_up
  )
  
  results <- rbind(results, data.frame(
    participant = p,
    param_estimate_mle = dclm$par,
    log_likelihood = -dclm$value
  ))
}
results <- results %>%
  mutate(param_estimate_mle = round(param_estimate_mle, 3))
results

results_MLE <- results %>%
  select(participant, param_estimate_mle)
results_MLE

################################################
############# Bayesian model ###################
################################################
dclmData <- read.csv('data_distractors.csv')

results_B <- data.frame(participant = integer(),
                      posterior_mean_B = numeric(),
                      stringsAsFactors = FALSE)

stan_model <- stan_model(file = 'bayesian.stan')

for (p in 1:22) {
  subj <- dclmData %>%
    filter(participant == p, condition == "distractors") %>%
    group_by(target_duration, distractors_duration_1, distractors_duration_2) %>%
    summarize(count_ones = sum(response == 1),
              ni = n()) %>%
    ungroup() %>%
    rename(mi = count_ones)
  
  td <- subj$target_duration
  dd1 <- subj$distractors_duration_1
  dd2 <- subj$distractors_duration_2
  mi <- subj$mi
  ni <- subj$ni
  
  data_list <- list(
    N = nrow(subj),
    mi = mi,
    ni = ni,
    td = td,
    dd1 = dd1,
    dd2 = dd2
  )
  
  # Fit the model
  fit <- stan(
    file = 'bayesian.stan',
    data = data_list,
    iter = 2000,
    chains = 4
  )
  
  # Extract parameter estimates
  param_samples <- rstan::extract(fit, pars = "param")$param
  param_est <- mean(param_samples)
  
  results_B <- rbind(results_B, data.frame(
    participant = p,
    posterior_mean_B = round(param_est, 3)
  ))
  
  # posterior distribution for each participant
  p_plot <- ggplot(data.frame(param = param_samples), aes(x = param)) +
    geom_density(fill = "blue", alpha = 0.5) +
    labs(title = paste("Posterior Distribution for Participant", p),
         x = "Parameter Estimate",
         y = "Density") +
    theme_minimal()
  
  print(p_plot)
}

results_B

#################################################
######### Hierarchical Bayesian model ###########
#################################################
dclmDataHB <- dclmData %>%
  filter(condition == "distractors") %>%
  group_by(participant, target_duration, distractors_duration_1, distractors_duration_2) %>%
  summarize(count_ones = sum(response == 1),
            ni = n()) %>%
  ungroup() %>%
  rename(mi = count_ones)

N <- nrow(dclmDataHB)
J <- length(unique(dclmDataHB$participant))
participant <- as.integer(factor(dclmDataHB$participant))
td <- dclmDataHB$target_duration
dd1 <- dclmDataHB$distractors_duration_1
dd2 <- dclmDataHB$distractors_duration_2
mi <- dclmDataHB$mi
ni <- dclmDataHB$ni

data_listHB <- list(
  N = N,
  mi = mi,
  ni = ni,
  td = td,
  dd1 = dd1,
  dd2 = dd2,
  J = J,
  participant = participant
)

stan_model_HB <- stan_model(file = 'hierarchicalBayesian.stan')

fitHB <- stan(
  file = 'hierarchicalBayesian.stan',
  data = data_listHB,
  iter = 2000,
  chains = 4
)

params <- rstan::extract(fitHB)
mu_param_mean <- mean(params$mu_param)
sigma_param_mean <- mean(params$sigma_param)

param_means <- apply(params$param, 2, mean)

results_HB <- data.frame(
  participant = 1:J,
  posterior_mean_HB = round(param_means, 3)
)

group_posterior_mean <- mean(param_means)
cat("Group posterior mean:", round(group_posterior_mean, 3), "\n")

results_HB

# Plot Group-level posterior distributions
mu_param_samples <- params$mu_param
sigma_param_samples <- params$sigma_param

mu_param_plot <- ggplot(data.frame(mu_param = mu_param_samples), aes(x = mu_param)) +
  geom_density(fill = "#31a354", alpha = 0.5) +
  labs(title = "Posterior Distribution of Group-level Mean (mu_param)",
       x = "mu_param",
       y = "Density") +
  theme_minimal()

sigma_param_plot <- ggplot(data.frame(sigma_param = sigma_param_samples), aes(x = sigma_param)) +
  geom_density(fill = "#31a354", alpha = 0.5) +
  labs(title = "Posterior Distribution of Group-level Std Dev (sigma_param)",
       x = "sigma_param",
       y = "Density") +
  theme_minimal()

mu_param_plot
sigma_param_plot

# Plot individual posterior distributions
for (p in 1:J) {
  param_samples <- params$param[, p]
  p_plot <- ggplot(data.frame(param = param_samples), aes(x = param)) +
    geom_density(fill = "blue", alpha = 0.5) +
    labs(title = paste("Posterior Distribution for Participant", p),
         x = "Parameter Estimate",
         y = "Density") +
    theme_minimal()
  
  print(p_plot)
}
##########################################
########## Visualize shrinkage ###########
##########################################

shrinkageData <- data.frame(
  participant = 1:22,
  posterior_mean_B = c(0.561, 0.723, 0.179, 0.400, 0.176, 0.703, 0.922, 0.218, 0.931, 0.651, 0.002, 0.611, 0.541, 0.018, 0.831, 0.520, 0.177, 0.083, 0.114, 0.166, 0.288, 0.010),
  posterior_mean_HB = c(0.550, 0.703, 0.207, 0.402, 0.186, 0.683, 0.899, 0.238, 0.912, 0.640, 0.002, 0.599, 0.534, 0.019, 0.810, 0.514, 0.193, 0.103, 0.130, 0.199, 0.304, 0.011)
)

group_mean <- 0.402

data_melt <- melt(shrinkageData, id.vars = "participant", variable.name = "model", value.name = "posterior_mean")

# Plot
ggplot(data_melt, aes(x = participant, y = posterior_mean, color = model)) +
  geom_point(size = 3) +
  geom_line(aes(group = participant), color = "grey") +
  geom_hline(aes(yintercept = group_mean, linetype = "Group Mean"), color = "red") +
  scale_color_manual(values = c("posterior_mean_B" = "blue", "posterior_mean_HB" = "green")) +
  scale_linetype_manual(name = "Legend", values = c("Group Mean" = "dashed")) +
  labs(title = "Shrinkage Effect in Hierarchical Bayesian Model",
       x = "Participant",
       y = "Posterior Mean",
       color = "Model") +
  scale_x_continuous(breaks = 1:22) +
  theme_minimal() +
  annotate("text", x = 0.5, y = group_mean, label = "0.402", color = "red", hjust = 0, vjust = -1)

#######################################
# Combine all results into one table
#######################################

all_results <- results_MLE %>%
  left_join(results_B, by = "participant") %>%
  left_join(results_HB, by = "participant")

all_results
