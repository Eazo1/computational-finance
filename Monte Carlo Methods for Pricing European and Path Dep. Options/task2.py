# Importing libraries
import math
import numpy as np
import scipy
from scipy.stats import norm, qmc
import matplotlib.pyplot as plt
import timeit

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

### Path Dependent Options

# Task 2.1
# Approcimate the path dependant Asian call option V using an Euler scheme
# Payoff function for the discrete fixed-strike Asian call option
def asian_call_option(S, X):
    return np.maximum(np.mean(S, axis=1) - X, 0)

# Euler scheme for the Monte Carlo simulation of the path dependent Asian call option
def euler_scheme_monte_carlo_asian_call_option(S0, X, T, r, alpha, beta, theta, sigma, gamma, n_paths, K, phi=None):
    dt = T / K # Time step

    # Sample iid random number from a normal distribution
    if phi is None:
        phi = rng.standard_normal((n_paths, K))
    else:
        phi = phi
    S = np.zeros((n_paths, K + 1))
    S[:, 0] = S0
    
    # Euler-Maruyama scheme for SDE
    for k in range(1, K + 1):
        S[:, k] = S[:, k - 1] + (alpha * theta - beta * S[:, k - 1]) * dt + sigma * np.power(np.abs(S[:, k - 1]), gamma) * np.sqrt(dt) * phi[:, k - 1]
    
    payoffs = asian_call_option(S[:, 1:], X) 
    
    # Discounted payoff average
    contract_value = np.exp(-r * T) * np.mean(payoffs)
    
    # Variance
    value_variance = np.var(np.exp(-r * T) * payoffs)
    
    return contract_value, value_variance

# Market fitted parameters
theta = 9478.91
alpha = 0.05
beta = 0.06
gamma = 0.93
sigma = 0.32 # Volatility of the option

r = 0.04 # Risk-free interest rate
S0 = 9494.205 # Stock price at t = 0
T = 1 # Maturity time
K = 65 # Number of equally spaced observations after the initial time
X = 9500 # Fixed strike price

sampler = qmc.Halton(d=K, scramble=True, seed=seed)

n_paths = 100000 # Number of paths

# Compute Option Price
option_price = euler_scheme_monte_carlo_asian_call_option(S0, X, T, r, alpha, beta, theta, sigma, gamma, n_paths, K)
print(f"Task 2.1, Estimated fixed-strike Asian call option price: {option_price}")

# Task 2.2
# Investigative Plots
# Investigate the convergence of the Euler scheme for the Monte Carlo simulation of the path dependent Asian call option
# by plotting the estimated option price as a function of the number of paths N

# Number of paths
#n_paths = np.logspace(1, 6, num=100, endpoint=True, base=10.0, dtype=int) # Computationally expensive
n_paths = np.logspace(1, 6, base=10.0, dtype=int) # Less computationally expensive

option_prices = np.zeros(len(n_paths))
variances = np.zeros(len(n_paths))


for i, n in enumerate(n_paths):
    option_prices[i], variances[i] = euler_scheme_monte_carlo_asian_call_option(S0, X, T, r, alpha, beta, theta, sigma, gamma, n, K)

plt.figure(figsize=(10, 6))
plt.plot(n_paths, option_prices, label='Estimated Option Price', color='blue',  linestyle='', markersize=1.5, marker='o')
plt.xscale('log')
plt.xlabel('Number of Paths')
plt.ylabel('Option Price')
plt.title('Monte Carlo Estimates of Euler-Maruyama Scheme Computed Asian Call Option Values')
plt.legend()
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.tight_layout()
plt.show()

# Task 2.2 Continuation
# Plot of estimate variance.

plt.figure(figsize=(10, 6))
plt.plot(n_paths, (variances / n_paths), label='Variance of Option Price Estimate', color='red', linestyle='', marker='o', markersize=1.5)
plt.xscale('log')
plt.xlabel('Number of Paths')
plt.ylabel('Variance')
plt.title('Variance of Monte Carlo Estimates of Asian Call Option Values')
plt.legend()
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.tight_layout()
plt.show()

# Task 2.3 
# Partial derivative value estimation
# Estimate the value of the partial derivative of the fixed-strike Asian call option with respect to gamma, at S0 and t=0
# using the forward finite difference method

def estimate_partial_derivative(S0, X, T, r, alpha, beta, theta, sigma, gamma, delta_gamma, n_paths, K, phi=None):
    
    phi = rng.standard_normal((n_paths, K))
    
    # Compute the option price at gamma
    option_price_gamma = euler_scheme_monte_carlo_asian_call_option(S0, X, T, r, alpha, beta, theta, sigma, gamma, n_paths, K, phi)[0]
    
    # Compute the option price at gamma + delta_gamma
    option_price_gamma_plus_delta_gamma = euler_scheme_monte_carlo_asian_call_option(S0, X, T, r, alpha, beta, theta, sigma, gamma + delta_gamma, n_paths, K, phi)[0]
    
    # Compute the partial derivative using the forward!! finite difference method
    partial_derivative = (option_price_gamma_plus_delta_gamma - option_price_gamma) / delta_gamma
    
    return partial_derivative

def estimate_partial_derivative_central(S0, X, T, r, alpha, beta, theta, sigma, gamma, delta_gamma, n_paths, K, phi=None):
    
    phi = rng.standard_normal((n_paths, K))
    
    option_price_gamma_plus = euler_scheme_monte_carlo_asian_call_option(S0, X, T, r, alpha, beta, theta, sigma, gamma + delta_gamma, n_paths, K, phi)[0]
    
    # Compute the option price at gamma - delta_gamma
    option_price_gamma_minus = euler_scheme_monte_carlo_asian_call_option(S0, X, T, r, alpha, beta, theta, sigma, gamma - delta_gamma, n_paths, K, phi)[0]
    
    # Compute the partial derivative using the central!!! finite difference method
    partial_derivative = (option_price_gamma_plus - option_price_gamma_minus) / (2 * delta_gamma)
    
    return partial_derivative

#######

def estimate_partial_derivative_halton(S0, X, T, r, alpha, beta, theta, sigma, gamma, delta_gamma, n_paths, K, phi=None):
    
    halton_samples = sampler.random(n_paths)
    phi = norm.ppf(halton_samples)
    
    # Compute the option price at gamma
    option_price_gamma = euler_scheme_monte_carlo_asian_call_option(S0, X, T, r, alpha, beta, theta, sigma, gamma, n_paths, K, phi)[0]
    
    # Compute the option price at gamma + delta_gamma
    option_price_gamma_plus_delta_gamma = euler_scheme_monte_carlo_asian_call_option(S0, X, T, r, alpha, beta, theta, sigma, gamma + delta_gamma, n_paths, K, phi)[0]
    
    # Compute the partial derivative using the forward!! finite difference method
    partial_derivative = (option_price_gamma_plus_delta_gamma - option_price_gamma) / delta_gamma
    
    return partial_derivative

def estimate_partial_derivative_halton_central(S0, X, T, r, alpha, beta, theta, sigma, gamma, delta_gamma, n_paths, K, phi=None):
    
    halton_samples = sampler.random(n_paths)
    phi = norm.ppf(halton_samples)
    
    option_price_gamma_plus = euler_scheme_monte_carlo_asian_call_option(S0, X, T, r, alpha, beta, theta, sigma, gamma + delta_gamma, n_paths, K, phi)[0]
    
    # Compute the option price at gamma - delta_gamma
    option_price_gamma_minus = euler_scheme_monte_carlo_asian_call_option(S0, X, T, r, alpha, beta, theta, sigma, gamma - delta_gamma, n_paths, K, phi)[0]
    
    # Compute the partial derivative using the central!!! finite difference method
    partial_derivative = (option_price_gamma_plus - option_price_gamma_minus) / (2 * delta_gamma)
    
    return partial_derivative

###

def estimate_partial_derivative_antithetic(S0, X, T, r, alpha, beta, theta, sigma, gamma, delta_gamma, n_paths, K, phi=None):
    
    if n_paths % 2 == 0:
        n_paths = n_paths
    else:
        n_paths = n_paths - 1
    
    phi_half = rng.standard_normal((n_paths // 2, K))
    phi = np.concatenate((phi_half, -phi_half), axis=0)
    
    # Compute the option price at gamma
    option_price_gamma = euler_scheme_monte_carlo_asian_call_option(S0, X, T, r, alpha, beta, theta, sigma, gamma, n_paths, K, phi)[0]
    
    # Compute the option price at gamma + delta_gamma
    option_price_gamma_plus_delta_gamma = euler_scheme_monte_carlo_asian_call_option(S0, X, T, r, alpha, beta, theta, sigma, gamma + delta_gamma, n_paths, K, phi)[0]
    
    # Compute the partial derivative using the forward!! finite difference method
    partial_derivative = (option_price_gamma_plus_delta_gamma - option_price_gamma) / delta_gamma
    
    return partial_derivative

def estimate_partial_derivative_moment_matching(S0, X, T, r, alpha, beta, theta, sigma, gamma, delta_gamma, n_paths, K, phi=None):
    
    if n_paths % 2 == 0:
        n_paths = n_paths
    else:
        n_paths = n_paths - 1
    
    phi_half = rng.standard_normal((n_paths // 2, K))
    phi = np.concatenate((phi_half, -phi_half), axis=0)
    row_var = np.var(phi, axis=1, keepdims=True)
    phi = phi / np.sqrt(row_var)
    
    # Compute the option price at gamma
    option_price_gamma = euler_scheme_monte_carlo_asian_call_option(S0, X, T, r, alpha, beta, theta, sigma, gamma, n_paths, K, phi)[0]
    
    # Compute the option price at gamma + delta_gamma
    option_price_gamma_plus_delta_gamma = euler_scheme_monte_carlo_asian_call_option(S0, X, T, r, alpha, beta, theta, sigma, gamma + delta_gamma, n_paths, K, phi)[0]
    
    # Compute the partial derivative using the forward!! finite difference method
    partial_derivative = (option_price_gamma_plus_delta_gamma - option_price_gamma) / delta_gamma
    
    return partial_derivative

###


gamma = 0.9299999999999999

# Tests to see if the code works

n_paths = 100000 # Number of paths
delta_gamma = 1e-4 # Step size for the finite difference method
#phi = rng.standard_normal((n_paths, K)) # Pre-generate random numbers

derivative_estimate = estimate_partial_derivative(S0, X, T, r, alpha, beta, theta, sigma, gamma, delta_gamma, n_paths, K)
print(f"Task 2.3, Estimated derivative dV/dγ: {derivative_estimate:.6f}")

derivative_estimate = estimate_partial_derivative_central(S0, X, T, r, alpha, beta, theta, sigma, 0.93, delta_gamma, n_paths, K)
print(f"Task 2.3, Estimated (central) derivative dV/dγ: {derivative_estimate:.6f}")

derivative_estimate = estimate_partial_derivative_halton(S0, X, T, r, alpha, beta, theta, sigma, gamma, delta_gamma, n_paths, K)
print(f"Task 2.3, Estimated (Halton) derivative dV/dγ: {derivative_estimate:.6f}")

derivative_estimate = estimate_partial_derivative_halton_central(S0, X, T, r, alpha, beta, theta, sigma, 0.93, delta_gamma, n_paths, K)
print(f"Task 2.3, Estimated (central, Halton) derivative dV/dγ: {derivative_estimate:.6f}")

###

# Plot of partial differential estimate vs n_paths, with both normal, antithetic, moment matching, Halton MC

n_paths = np.logspace(3, 6, 10, base=10, dtype=int)
values_normal = [estimate_partial_derivative(S0, X, T, r, alpha, beta, theta, sigma, gamma, delta_gamma, n, K) for n in n_paths]
values_halton = [estimate_partial_derivative_halton(S0, X, T, r, alpha, beta, theta, sigma, gamma, delta_gamma, n, K) for n in n_paths]
values_antithetic = [estimate_partial_derivative_antithetic(S0, X, T, r, alpha, beta, theta, sigma, gamma, delta_gamma, n, K) for n in n_paths]
values_moment_matching = [estimate_partial_derivative_moment_matching(S0, X, T, r, alpha, beta, theta, sigma, gamma, delta_gamma, n, K) for n in n_paths]

plt.figure(figsize=(10,6))
plt.plot(n_paths, values_normal, color='blue', marker='o', linestyle='-', label='Normal MC (Forward FD)')
plt.plot(n_paths, values_halton, color='red', marker='o', linestyle='-', label='Halton MC (Forward FD)')
plt.plot(n_paths, values_antithetic, color='green', marker='o', linestyle='-', label='Antithetic Sampling MC (Forward FD)')
plt.plot(n_paths, values_moment_matching, color='orange', marker='o', linestyle='-', label='Moment Matching MC (Forward FD)')
plt.xscale('log')
plt.xlabel('Number of Paths')
plt.ylabel('dV/dγ')
plt.title('dV/dγ vs. n_paths (Various MC Techniques)')
plt.legend()
plt.grid(True, which='both', linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.show()

# Print Last Value obtained from Halton MC
print(f"Task 2.3, Final Estimated (Halton) derivative dV/dγ: {values_halton[-1]:.6f}")

# Plot of partial differential estimate vs n_paths, with both forward and central finite difference, both normal and halton
n_paths = np.logspace(2, 4, 60, base=10, dtype=int)
#values_normal = [estimate_partial_derivative(S0, X, T, r, alpha, beta, theta, sigma, gamma, delta_gamma, n, K) for n in n_paths]
values_halton = [estimate_partial_derivative_halton(S0, X, T, r, alpha, beta, theta, sigma, gamma, delta_gamma, n, K) for n in n_paths]
#values_normal_central = [estimate_partial_derivative_central(S0, X, T, r, alpha, beta, theta, sigma, 0.93, delta_gamma, n, K) for n in n_paths]
values_halton_central = [estimate_partial_derivative_halton_central(S0, X, T, r, alpha, beta, theta, sigma, 0.93, delta_gamma, n, K) for n in n_paths]

plt.figure(figsize=(10,6))
#plt.plot(n_paths, values_normal, color='blue', marker='o', markersize = 1.1, linestyle='--', linewidth = 1, label='Normal MC (Forward FD)')
plt.plot(n_paths, values_halton, color='red', marker='o', markersize = 1.1, linestyle='--', linewidth = 1, label='Halton MC (Forward FD)')
#plt.plot(n_paths, values_normal_central, color='green', marker='o', markersize = 1.1, linestyle='--', linewidth = 1, label='Normal MC (Central FD)')
plt.plot(n_paths, values_halton_central, color='orange', marker='o', markersize = 1.1, linestyle='--', linewidth = 1, label='Halton MC (Central FD)')
plt.xscale('log')
plt.xlabel('Number of Paths')
plt.ylabel('dV/dγ')
plt.title('dV/dγ vs. n_paths (Central vs Forward Finite Difference)')
plt.legend()
plt.grid(True, which='both', linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.show()

# Again but with larger delta gamma
delta_gamma = 1e-2
# Plot of partial differential estimate vs n_paths, with both forward and central finite difference, both normal and halton
n_paths = np.logspace(2, 4, 60, base=10, dtype=int)
#values_normal = [estimate_partial_derivative(S0, X, T, r, alpha, beta, theta, sigma, gamma, delta_gamma, n, K) for n in n_paths]
values_halton = [estimate_partial_derivative_halton(S0, X, T, r, alpha, beta, theta, sigma, gamma, delta_gamma, n, K) for n in n_paths]
#values_normal_central = [estimate_partial_derivative_central(S0, X, T, r, alpha, beta, theta, sigma, 0.93, delta_gamma, n, K) for n in n_paths]
values_halton_central = [estimate_partial_derivative_halton_central(S0, X, T, r, alpha, beta, theta, sigma, 0.93, delta_gamma, n, K) for n in n_paths]

plt.figure(figsize=(10,6))
#plt.plot(n_paths, values_normal, color='blue', marker='o', markersize = 1.1, linestyle='--', linewidth = 1, label='Normal MC (Forward FD)')
plt.plot(n_paths, values_halton, color='red', marker='o', markersize = 1.1, linestyle='--', linewidth = 1, label='Halton MC (Forward FD)')
#plt.plot(n_paths, values_normal_central, color='green', marker='o', markersize = 1.1, linestyle='--', linewidth = 1, label='Normal MC (Central FD)')
plt.plot(n_paths, values_halton_central, color='orange', marker='o', markersize = 1.1, linestyle='--', linewidth = 1, label='Halton MC (Central FD)')
plt.xscale('log')
plt.xlabel('Number of Paths')
plt.ylabel('dV/dγ')
plt.title('dV/dγ vs. n_paths (Central vs Forward Finite Difference), Larger Δγ')
plt.legend()
plt.grid(True, which='both', linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.show()

# Plot of partial differential estimate n_paths=100000 vs delta_gamma (step size), normal MC
n_paths = 100000 # Number of paths
delta_gamma = np.logspace(0, -4, 25, base=10)

values_normal = [estimate_partial_derivative(S0, X, T, r, alpha, beta, theta, sigma, gamma, dg, n_paths, K) for dg in delta_gamma]

plt.figure(figsize=(10,6))
plt.plot(delta_gamma, values_normal, color='blue', marker='o', markersize = 1.4, linestyle='-', linewidth = 1, label='Partial Differential Estimate')
plt.xscale('log')
plt.gca().invert_xaxis()
plt.xlabel('Δγ Finite Difference Step')
plt.ylabel('dV/dγ')
plt.title('dV/dγ vs. Δγ')
plt.legend()
plt.grid(True, which='both', linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.show()
