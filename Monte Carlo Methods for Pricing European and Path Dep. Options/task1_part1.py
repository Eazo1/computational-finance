# Importing libraries
import math
import numpy as np
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt

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

rng = np.random.default_rng(seed=123)

### The Simple European Financial Contract

# Functions describing the distribution of the stock price at T
f = lambda S0, T, alpha, beta, theta: S0 * np.log(1 + alpha * T - beta * T) + theta * np.cosh(2 * beta * T - alpha * T)

v = lambda S0, T, sigma, alpha, gamma, theta: sigma * (np.square(1 + alpha * T)) * np.power(S0, (3 * gamma)) * np.power(theta, (-2 * gamma))


# Function defining the payoff on expiry for the simple European contract
def european_payoff_on_expiry(ST, X1, X2):
    return np.piecewise(ST, [ST < X1, (ST >= X1) & (ST < X2), ST >= X2], [lambda ST: ST, lambda ST: X1 - X2, lambda ST: X2 - X1])

# Task 1.1
# Monte-Carlo Simulation
def monte_carlo_contract_value(S0, X1, X2, T, r, alpha, beta, theta, sigma, gamma, n_paths):

    # Sample iid random number from a normal distribution
    phi = rng.standard_normal(n_paths) # #n_paths samples from a normal distribution
    ST = f(S0, T, alpha, beta, theta) + v(S0, T, sigma, alpha, gamma, theta) * np.sqrt(T) * phi
    
    # Summing payoffs
    payoff = np.sum(european_payoff_on_expiry(ST, X1, X2))

    # Discounted payoff average
    contract_value = (np.exp(-r * T) * (1/n_paths)) * payoff
    
    return contract_value

# Using the analytic equation for the payoff, and using quadrature methods for numerical integration
def A_func(r,  v, T):
    return np.exp(-r * T) / (v * np.sqrt(2 * np.pi * T))

# def B_func(z,  f,  v, T):
#     exponent = (-(z - f) * (z - f)) / (2. * v * v * T)
#     return np.exp(exponent)


def integrand(z, T, f, v, X1, X2):
    payoff = european_payoff_on_expiry(z, X1, X2)
    B_func = np.exp(-(np.square(z - f)) / (2 * np.square(v) * T))
    return payoff * B_func

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

# Monte Carlo

n_paths = 100000
value1 = monte_carlo_contract_value(S0, X1, X2, T, r, alpha, beta, theta, sigma, gamma, n_paths)
print(f'Task 1, Estimated Contract Value (Monte-Carlo, N = {n_paths}): {value1}')


# Analytic Solution
# Limit integration range to 5 standard deviations from the mean in both directions to avoid numerical issues
lower_bound = f(S0, T, alpha, beta, theta) - 5 * v(S0, T, sigma, alpha, gamma, theta) * np.sqrt(T)
upper_bound = f(S0, T, alpha, beta, theta) + 5 * v(S0, T, sigma, alpha, gamma, theta) * np.sqrt(T)

integral_result, integral_error = scipy.integrate.quad(integrand, lower_bound, upper_bound, args=(T, f(S0, T, alpha, beta, theta), v(S0, T, sigma, alpha, gamma, theta), X1, X2), limit=100)
result = A_func(r, v(S0, T, sigma, alpha, gamma, theta), T) * integral_result
error = A_func(r, v(S0, T, sigma, alpha, gamma, theta), T) * integral_error
print(f'Task 1, Estimated Contract Value (Quadrature): {result} + {error}')

# Task 1.2
# Figure showing Monte Carlo approximate of contract value C(S0, t = 0) with increasing n, alongside exact value from the analytical formula
#n_paths = np.linspace(1000, 1000000, 1000, dtype=int)
n_paths = np.logspace(3, 7, 1000, base=10, dtype=int)
values = [monte_carlo_contract_value(S0, X1, X2, T, r, alpha, beta, theta, sigma, gamma, n) for n in n_paths]

#plot using rcparams

plt.figure(figsize=(8, 6))
plt.plot(n_paths, values, label='Monte Carlo Estimates', linestyle='', marker='o', color='black', markersize=0.8)
plt.axhline(y=result, color='red', label='Analytic Value')
# Analytic value error as a shaded region
plt.fill_between(n_paths, result - error, result + error, color='red', alpha=0.2)

plt.xscale("log")
plt.xlabel('Number of Paths')
plt.ylabel('Contract Value')
plt.title('Monte Carlo Estimates of Contract Value vs Number of Paths')
plt.minorticks_on()
plt.legend(frameon=True, loc='best')
plt.grid(True, linestyle='dashed', linewidth=0.5)
plt.show()

# # Same plot but binning the Monte Carlo estimates, with error bars from variance
# n_paths = np.linspace(1000, 1000000, 1000, dtype=int)
# values = [monte_carlo_contract_value(S0, X1, X2, T, r, alpha, beta, theta, sigma, gamma, n) for n in n_paths]

# # Bin the values
# n_bins = 100
# bin_size = len(values) // n_bins
# binned_values = np.mean(np.array(values).reshape(-1, bin_size), axis=1)
# binned_errors = np.std(np.array(values).reshape(-1, bin_size), axis=1)

# plt.figure(figsize=(8, 6))
# plt.errorbar(n_paths[::bin_size], binned_values, yerr=binned_errors, label='Monte Carlo Estimates', linestyle='', marker='o', color='black', markersize=0.8)
# plt.axhline(y=result, color='red', label='Analytic Value')
# # Analytic value error as a shaded region
# plt.fill_between(n_paths, result - error, result + error, color='red', alpha=0.2)

# plt.xlabel('Number of Paths')
# plt.ylabel('Contract Value')
# plt.title('Monte Carlo Estimates of Contract Value vs Number of Paths')
# plt.minorticks_on()
# plt.legend(frameon=True, loc='best')
# plt.grid(True, linestyle='dashed', linewidth=0.5)
# plt.show()
