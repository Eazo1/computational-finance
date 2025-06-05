# Importing libraries
import math
import numpy as np
import scipy
from scipy.stats import norm, qmc
import matplotlib.pyplot as plt
import timeit


# Functions describing the distribution of the stock price at T
f = lambda S0, T, alpha, beta, theta: S0 * np.log(1 + alpha * T - beta * T) + theta * np.cosh(2 * beta * T - alpha * T)

v = lambda S0, T, sigma, alpha, gamma, theta: sigma * (np.square(1 + alpha * T)) * np.power(S0, (3 * gamma)) * np.power(theta, (-2 * gamma))

def european_payoff_on_expiry(ST, X1, X2):
    return np.piecewise(ST, [ST < X1, (ST >= X1) & (ST < X2), ST >= X2], [lambda ST: ST, lambda ST: X1 - X2, lambda ST: X2 - X1])

# Latex style for plots
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

seed = 123
rng = np.random.default_rng(seed=seed)
sampler = qmc.Halton(d=1, scramble=True, seed=seed)

# General information
S0 = 7492.43 # Stock price at t = 0
r = 0.02 # Risk-free interest rate
T = 1.75 # Maturity time
X1 = 7500 # Strike price 1
X2 = 8500 # Strike price 2

# Market fitted parameters
theta = 7491
alpha = 0.03
beta = 0.04
gamma = 0.91
sigma = 0.29 # Volatility of the option


### The Simple European Financial Contract

# Task 1.3
# Using and comparing different random number generators: antithetic variables, moment matching, Halton sequences

# Antithetic variables
def monte_carlo_antithetic_variables(S0, X1, X2, T, r, alpha, beta, theta, sigma, gamma, n_paths):

    # Sample iid random number from a normal distribution
    phi = rng.standard_normal(n_paths // 2)
    phi = np.concatenate((phi, -phi))
    ST = f(S0, T, alpha, beta, theta) + v(S0, T, sigma, alpha, gamma, theta) * np.sqrt(T) * phi
    
    # Summing payoffs
    payoffs = european_payoff_on_expiry(ST, X1, X2)
    payoff = np.sum(payoffs)

    # Discounted payoff average
    contract_value = (np.exp(-r * T) * (1/n_paths)) * payoff
        
    return contract_value

# Moment matching
def monte_carlo_moment_matching(S0, X1, X2, T, r, alpha, beta, theta, sigma, gamma, n_paths):

    # Sample iid random number from a normal distribution
    phi = rng.standard_normal(n_paths // 2)
    phi = np.concatenate((phi, -phi))
    var = np.var(phi)
    phi = (1/np.sqrt(var)) * phi
    ST = f(S0, T, alpha, beta, theta) + v(S0, T, sigma, alpha, gamma, theta) * np.sqrt(T) * phi
    
    # Summing payoffs
    payoffs = european_payoff_on_expiry(ST, X1, X2)
    payoff = np.sum(payoffs)

    # Discounted payoff average
    contract_value = (np.exp(-r * T) * (1/n_paths)) * payoff
    
    return contract_value

# Halton sequences
def monte_carlo_halton_sequences(S0, X1, X2, T, r, alpha, beta, theta, sigma, gamma, n_paths):

    halton_samples = sampler.random(n_paths)
    
    phi = norm.ppf(halton_samples)
    ST = f(S0, T, alpha, beta, theta) + v(S0, T, sigma, alpha, gamma, theta) * np.sqrt(T) * phi
    
    # Summing payoffs
    payoffs = european_payoff_on_expiry(ST, X1, X2)
    payoff = np.sum(payoffs)

    # Discounted payoff average
    contract_value = (np.exp(-r * T) * (1/n_paths)) * payoff
        
    return contract_value

# Standard (for comparison, no variance reduction)
def monte_carlo_standard(S0, X1, X2, T, r, alpha, beta, theta, sigma, gamma, n_paths):

    # Sample iid random number from a normal distribution
    phi = rng.standard_normal(n_paths)
    ST = f(S0, T, alpha, beta, theta) + v(S0, T, sigma, alpha, gamma, theta) * np.sqrt(T) * phi
    
    # Summing payoffs
    payoffs = european_payoff_on_expiry(ST, X1, X2)
    payoff = np.sum(payoffs)

    # Discounted payoff average
    contract_value = (np.exp(-r * T) * (1/n_paths)) * payoff
        
    return contract_value

# Investigate how the accuracy of the approximation changes with different value of n.
n_values = np.logspace(3, 7, 10, base=10, dtype=int)
n_repeats = 10

# Monte Carlo simulation
monte_carlo_functions = [monte_carlo_antithetic_variables, monte_carlo_moment_matching, monte_carlo_halton_sequences, monte_carlo_standard]
monte_carlo_labels = ["Antithetic Variables", "Moment Matching", "Halton Sequences", "Standard MC"]

# Monte Carlo simulation Values and Plotting
fig, ax = plt.subplots(figsize=(10, 6))
# Comparing accuracy of the approximation with different value of n
for i, monte_carlo_function in enumerate(monte_carlo_functions):
    mean_values = []
    for n in n_values:
        values = [monte_carlo_function(S0, X1, X2, T, r, alpha, beta, theta, sigma, gamma, n) for _ in range(n_repeats)]
        mean_values.append(np.mean(values))
    ax.plot(n_values, mean_values, label=monte_carlo_labels[i], marker="o", linestyle="-")

    
ax.set_xscale("log")
ax.set_xlabel("Number of Paths")
ax.set_ylabel("Contract Value")
ax.set_title("Monte Carlo Estimates of Contract Value by Path Count and Method")
ax.minorticks_on()
ax.axhline(y=3211.1167122553197, color='red', label='Analytic Value')
ax.grid(True, linestyle='dashed', linewidth=0.5)
ax.legend(frameon=True, loc='best')
plt.show()

