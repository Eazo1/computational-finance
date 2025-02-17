# Importing libraries
import math
from scipy.stats import norm
import numpy as np
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

# Defining relevant function arguments seperately
m = lambda r, t, T, kappa, theta: r*math.exp(-kappa*(T-t)) + theta*(1-math.exp(-kappa*(T-t)))

n = lambda r, t, T, kappa, theta: 0.5*r*(T-t) - ((theta-r)/kappa)*(1-math.exp(-3*kappa*(T-t)))

k_squared = lambda t, T, kappa, sigma: ((sigma**2)/(2*(kappa**3)))*(5*math.exp(-kappa*(T-t)) - 3*math.exp(-2*kappa*(T-t)) + 3*kappa*(T-t) - 2)

q  = lambda t, T, kappa, sigma: ((sigma**2)/(4*(kappa**2)))*((1-math.exp(-kappa*(T-t)))**3)


# Defining function for pure discount bond price
def P(r, t, T, kappa, theta, sigma):
  return math.exp(0.25*k_squared(t, T, kappa, sigma) - 0.5*n(r, t, T, kappa, theta))

# Defining function for mean of R under risk-neutral measure
def f(r, t, T, kappa, theta, sigma):
  return m(r, t, T, kappa, theta) - 0.5*q(t, T, kappa, sigma)

# Defining function for variance of R under risk-neutral measure
def v_squared(t, T, kappa, sigma):
  return ((sigma**2)/(3*kappa))*(1-math.exp(-3*kappa*(T-t)))

# Defining function for h argument in the Call and Put option pricing formulas
def h(X_r, r, t, T, kappa, theta, sigma):
  return (X_r-f(r, t, T, kappa, theta, sigma))/(v_squared(t, T, kappa, sigma)**0.5)

# Defining function for Put option price
def put_option_price(X_r, r, t, T, kappa, theta, sigma):
  return P(r, t, T, kappa, theta, sigma)*norm.cdf(h(X_r, r, t, T, kappa, theta, sigma))

#####
# Main to run the program

def main():
  # Evaluation of example put option price using the given parameters/constants
  kappa_val = 0.2848
  theta_val = 0.067
  sigma_val = 0.0378
  
  task_2_put_option_price = put_option_price(X_r=0.05, r=0.0283, t=0, T=8, kappa=kappa_val, theta=theta_val, sigma=sigma_val)
  
  print('Task 2: Put option price: %.6f' % task_2_put_option_price)
  
  ###
  # Put option and bond prices for different values of r
  option_price_array = np.array([])
  bond_price_array = np.array([])
  
  print('Task 3: Put option and bond prices for different values of r:')
  
  for r in np.linspace(0, 0.2, 100):
    option_price_r_i = put_option_price(X_r=0.05, r=r, t=0, T=8, kappa=kappa_val, theta=theta_val, sigma=sigma_val)
    bond_price_i = P(r=r, t=0, T=8, kappa=kappa_val, theta=theta_val, sigma=sigma_val)
    
    print('r: %.4f, Put option price: %.4f, Bond price: %.4f' % (r, option_price_r_i, bond_price_i))
    
    option_price_array = np.append(option_price_array, option_price_r_i)
    bond_price_array = np.append(bond_price_array, bond_price_i)
  
  # Plotting the put option and bond prices for different values of r, and saving the plot as a PDF
  # Clean style for the plot
  plt.figure(figsize=(6, 4))
  plt.plot(np.linspace(0, 0.2, 100), option_price_array, label=r'$V(r, t = 0, T)$', color='blue', linewidth=1.5)
  plt.plot(np.linspace(0, 0.2, 100), bond_price_array, label=r'$P(r, t = 0, T)$', color='red', linewidth=1.5)
  plt.xlabel(r'Interest Rate $r$')
  plt.ylabel(r'$\$  \text{Price}$')
  plt.ylim(0)
  plt.xlim(0, 0.2)
  plt.title(r'Relevant Put Option and Bond Prices for $r \in [0, 0.2]$')
  plt.legend(frameon=True, loc='best')
  plt.grid(True, linestyle='dashed', linewidth=0.5)
  plt.savefig('task_3_plot.pdf', dpi=300, bbox_inches='tight')
  plt.show()

# Run main function
if __name__ == '__main__':
  main()