# Importing libraries
import math
import numpy as np
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from math import exp

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

### Finite Difference Methods Assignment
## Task 2.1 Bonds
# Part b

# General information and market-fitted parameters
T = 3  # Maturity time
F = 81 
theta = 0.0262
r0 = 0.0381
kappa = 0.08169
mu = 0.015
C = 2.58
alpha = 0.02
beta = 0.413
sigma = 0.111

# Crank-Nicolson method for the bond pricing PDE
# Crank-Nicolson Scheme Parameters

iMax = 100 # max number of time steps
jMax = 100 # max number of space steps

rMax = 1.0 # max interest rate
dr = rMax / jMax  # r step size
dt = T / iMax # time step size

# Numpy arrays for storing values
r = np.zeros(jMax+1)
t = np.zeros(iMax+1)
B_new = np.zeros(jMax+1)
B_old = np.zeros(jMax+1)

for i in range(iMax+1):
    t[i] = i*dt

for j in range(jMax+1):
    r[j] = j*dr

# Record the value of the bond at maturity
B_old[:] = F

# Matrix solution for the Crank-Nicolson scheme
# Storage for A (tridiagonal matrix)
A_bands = np.zeros(shape=(3,jMax+1))

band_structure = (1, 1) # (lower_bands, upper_bands)

# Allocate storage for the RHS term (d)
d = np.zeros(jMax+1) 

# Loop over time steps
for i in range(iMax-1, -1, -1):

    # Fill in the tridiagonal matrix A
    # Clear the A_bands array for each time step
    A_bands.fill(0.0)

    # Special case for j = 0 (PDE boundary condition at r = 0)
    common_part_bc = (kappa * theta * ( np.exp(mu*i*dt) + np.exp(mu*(i+1)*dt)))/(2.0*dr)
    A_bands[1,0] = -1.0/dt - common_part_bc # b_0
    A_bands[0,1] =  common_part_bc # c_0

    # a[j], b[j], c[j] for matrix middle rows
    for j in range(1, jMax):
        
        # Terms for convenience
        j_term = (j**(2*beta)) * (dr**(2*beta - 2))
        k_part = kappa * (theta * (np.exp(mu*i*dt) + np.exp(mu*(i+1)*dt)) - j*dr)

        # a_j B_j-1 + b_j B_j + c_i B_j+1 = d_i
        A_bands[2, j-1] = 0.25 * sigma**2 * j_term  -  (k_part / (4*dr)) # a_j
        A_bands[1, j] = -1.0/dt  - 0.5 * sigma**2 * j_term  - 0.5 * j * dr # b_j
        A_bands[0, j+1] = 0.25 * sigma**2 * j_term  +  (k_part / (4*dr)) # c_j


    # Boundary condition at for j = 0 jMax
    # a_jMax B_jMax-1 + b_jMax B_jMax = d_jMax
    A_bands[2,jMax-1] = 0.0 # a_jMax
    A_bands[1,jMax] = 1.0 # b_jMax
    
    # Fill in the RHS term (d) for the Crank-Nicolson scheme
    # Clear the d array for each time step
    d.fill(0.0)

    # Common term
    #exp_term = 0.5*C*np.exp(-alpha*t[i]) + 0.5*C*np.exp(-alpha*t[i+1])
    exp_term = 0.5*C*np.exp(-alpha*(t[i]+0.5*dt))

    # Special case for j = 0 (PDE boundary condition at r = 0)
    c0 = (kappa*theta*(np.exp(mu*t[i]) + np.exp(mu*t[i+1])))/(2.0*dr)
    d[0] = (-1.0/dt + c0) * B_old[0] - c0 * B_old[1] - exp_term

    # Case for 1 <= j < jMax
    for j in range(1, jMax):
        aa = A_bands[2, j-1]
        bb = A_bands[1, j] + 2.0/dt
        cc = A_bands[0, j+1]

        d[j] = -aa * B_old[j-1] - bb * B_old[j] - cc * B_old[j+1] - exp_term
        
    # Case for j = jMax
    d[jMax] = 0.0 # d_jMax

    # Solve the equation
    B_new = solve_banded(band_structure, A_bands, d)
    B_old = np.copy(B_new)

print(B_new)

B_r0 = np.interp(r0, r, B_new)
print("Bond value at r0, t=0  :", B_r0)


# Part c
# Function to compute the bond value using the Crank-Nicolson method with Dirichlet boundary conditions
def crank_dirichlet(A_bands, d, band_structure, B_old):
    for i in range(iMax-1, -1, -1):

        # Fill in the tridiagonal matrix A
        A_bands.fill(0.0)

        # Special case for j = 0 (PDE boundary condition at r = 0)
        common_part_bc = (kappa * theta *
                          (np.exp(mu*i*dt) + np.exp(mu*(i+1)*dt))) \
                         / (2.0*dr)
        A_bands[1,0] = -1.0/dt - common_part_bc   # b_0
        A_bands[0,1] =  common_part_bc            # c_0

        # a[j], b[j], c[j] for matrix middle rows
        for j in range(1, jMax):
            j_term = (j**(2*beta)) * (dr**(2*beta - 2))
            k_part = kappa * (
                theta*(np.exp(mu*i*dt) + np.exp(mu*(i+1)*dt)) - j*dr
            )
            A_bands[2,j-1] = 0.25*sigma**2*j_term - k_part/(4*dr)  # a_j
            A_bands[1,j  ] = -1.0/dt - 0.5*sigma**2*j_term - 0.5*j*dr  # b_j
            A_bands[0,j+1] = 0.25*sigma**2*j_term + k_part/(4*dr)  # c_j

        # Boundary condition at for j = 0 jMax
        # a_jMax B_jMax-1 + b_jMax B_jMax = d_jMax
        A_bands[2,jMax-1] = 0.0   # a_jMax
        A_bands[1,jMax  ] = 1.0   # b_jMax

        # Fill in the RHS term (d) for the Crank-Nicolson scheme
        d.fill(0.0)
        #exp_term = 0.5*C*np.exp(-alpha*t[i]) + 0.5*C*np.exp(-alpha*t[i+1])
        exp_term = 0.5*C*np.exp(-alpha*(t[i]+0.5*dt))

        c0 = common_part_bc
        d[0] = (-1.0/dt + c0)*B_old[0] - c0*B_old[1] - exp_term

        for j in range(1, jMax):
            aa = A_bands[2,j-1]
            bb = A_bands[1,j] + 2.0/dt
            cc = A_bands[0,j+1]
            d[j] = -aa*B_old[j-1] - bb*B_old[j] - cc*B_old[j+1] - exp_term

        d[jMax] = 0.0   # d_jMax

        # Solve the equation
        B_old = solve_banded(band_structure, A_bands, d)

    return B_old

# Function to compute the bond value using the Crank-Nicolson method with Neumann boundary conditions
def crank_neumann(A_bands, d, band_structure, B_old):
    for i in range(iMax-1, -1, -1):

        # Fill in the tridiagonal matrix A
        A_bands.fill(0.0)

        # Special case for j = 0 (PDE boundary condition at r = 0)
        common_part_bc = (kappa * theta *
                          (np.exp(mu*i*dt) + np.exp(mu*(i+1)*dt))) \
                         / (2.0*dr)
        A_bands[1,0] = -1.0/dt - common_part_bc
        A_bands[0,1] =  common_part_bc

        # a[j], b[j], c[j] for matrix middle rows
        for j in range(1, jMax):
            j_term = (j**(2*beta)) * (dr**(2*beta - 2))
            k_part = kappa * (
                theta*(np.exp(mu*i*dt) + np.exp(mu*(i+1)*dt)) - j*dr
            )
            A_bands[2,j-1] = 0.25*sigma**2*j_term - k_part/(4*dr)
            A_bands[1,j  ] = -1.0/dt - 0.5*sigma**2*j_term - 0.5*j*dr
            A_bands[0,j+1] = 0.25*sigma**2*j_term + k_part/(4*dr)

        # Neumann row: -B_{jMax-1} + B_{jMax} = 0
        A_bands[2,jMax-1] = -1.0   # a_jMax
        A_bands[1,jMax  ] =  1.0   # b_jMax

        # Fill in the RHS term (d) for the Crank-Nicolson scheme
        d.fill(0.0)
        #exp_term = 0.5*C*np.exp(-alpha*t[i]) + 0.5*C*np.exp(-alpha*t[i+1])
        exp_term = 0.5*C*np.exp(-alpha*(t[i]+0.5*dt))

        c0 = common_part_bc
        d[0] = (-1.0/dt + c0)*B_old[0] - c0*B_old[1] - exp_term

        for j in range(1, jMax):
            aa = A_bands[2,j-1]
            bb = A_bands[1,j] + 2.0/dt
            cc = A_bands[0,j+1]
            d[j] = -aa*B_old[j-1] - bb*B_old[j] - cc*B_old[j+1] - exp_term

        d[jMax] = 0.0   # d_jMax

        # Solve the equation
        B_old = solve_banded(band_structure, A_bands, d)

    return B_old

# Runs for Comparison Plots

iMax = 100 # max number of time steps
jMax = 100 # max number of space steps

rMax = 1.0 # max interest rate
dr = rMax / jMax  # r step size
dt = T / iMax # time step size

# Numpy arrays for storing values
r = np.zeros(jMax+1)
t = np.zeros(iMax+1)
B_new = np.zeros(jMax+1)
B_old = np.zeros(jMax+1)

for i in range(iMax+1):
    t[i] = i*dt

for j in range(jMax+1):
    r[j] = j*dr
    
A_bands = np.zeros(shape=(3,jMax+1))

band_structure = (1, 1) # (lower_bands, upper_bands)

# Allocate storage for the RHS term (d)
d = np.zeros(jMax+1) 

# Dirichlet run:
B_old[:] = F
B_dirich = crank_dirichlet(A_bands, d, band_structure, B_old.copy())

B_r0_Dirichlet = np.interp(r0, r, B_dirich)
print("Bond value at r0, t=0  Dirichlet:", B_r0_Dirichlet)

# Neumann run (start again from maturity):
B_old[:] = F
B_neum  = crank_neumann(A_bands, d, band_structure, B_old.copy())

B_r0_Neumann = np.interp(r0, r, B_neum)
print("Bond value at r0, t=0  Neumann:", B_r0_Neumann)

# Plotting the results, comparing Dirichlet and Neumann (0 to rMax)

plt.figure(figsize=(10, 6))
plt.plot(r, B_dirich, label='Dirichlet', color='blue', linestyle='--')
plt.plot(r, B_neum, label='Neumann', color='red')
plt.title('Bond Value Comparison: Dirichlet vs Neumann')
plt.xlabel('Interest Rate (r)')
plt.ylabel('Bond Value')
plt.legend()
plt.grid()
plt.xlim(0, rMax)
#plt.ylim(0, 100)
plt.show()

# Code for Extra
# Neumann
rMax = 5.0
dr = rMax/jMax
dt = T/iMax
r_neu = np.linspace(0, rMax, jMax+1)
B_old = np.full(jMax+1, F)
A_bands = np.zeros((3, jMax+1))
d = np.zeros(jMax+1)
B_neu = crank_neumann(A_bands, d, (1,1), B_old)

# Dirichlet rMax=1
rMax = 1.0
dr = rMax/jMax
r1 = np.linspace(0, rMax, jMax+1)
B_old = np.full(jMax+1, F)
B_dir1 = crank_dirichlet(A_bands, d, (1,1), B_old)
B_dir1_i = np.interp(r_neu, r1, B_dir1)

# Dirichlet rMax=2
rMax = 2.0
dr = rMax/jMax
r2 = np.linspace(0, rMax, jMax+1)
B_old = np.full(jMax+1, F)
B_dir2 = crank_dirichlet(A_bands, d, (1,1), B_old)
B_dir2_i = np.interp(r_neu, r2, B_dir2)

# Dirichlet rMax=3
rMax = 3.0
dr = rMax/jMax
r3 = np.linspace(0, rMax, jMax+1)
B_old = np.full(jMax+1, F)
B_dir3 = crank_dirichlet(A_bands, d, (1,1), B_old)
B_dir3_i = np.interp(r_neu, r3, B_dir3)

# Dirichlet rMax=4
rMax = 4.0
dr = rMax/jMax
r4 = np.linspace(0, rMax, jMax+1)
B_old = np.full(jMax+1, F)
B_dir4 = crank_dirichlet(A_bands, d, (1,1), B_old)
B_dir4_i = np.interp(r_neu, r4, B_dir4)

# Dirichlet rMax=5
rMax = 5.0
dr = rMax/jMax
r5 = np.linspace(0, rMax, jMax+1)
B_old = np.full(jMax+1, F)
B_dir5 = crank_dirichlet(A_bands, d, (1,1), B_old)
B_dir5_i = np.interp(r_neu, r5, B_dir5)

plt.figure(figsize=(10, 6))
plt.plot(r_neu, B_neu, label='Neumann', color='red', linewidth=3, alpha=0.7)
plt.plot(r_neu, B_dir1_i, label='Dirichlet', color='blue', linestyle='--', linewidth=1)
plt.plot(r_neu, B_dir2_i, color='blue', linestyle='--', linewidth=1)
plt.plot(r_neu, B_dir3_i, color='blue', linestyle='--', linewidth=1)
plt.plot(r_neu, B_dir4_i, color='blue', linestyle='--', linewidth=1)
plt.plot(r_neu, B_dir5_i, color='blue', linestyle='--', linewidth=1)
plt.title('Bond Value Comparison: Dirichlet vs Neumann, Varying Dirichlet rMax')
plt.xlabel('Interest Rate (r)')
plt.ylabel('Bond Value')
plt.legend()
plt.grid()
plt.xlim(0, 4)
plt.ylim(0, 20)
plt.show()


#######################################################

## Task 2.2 Options on Bonds (American Call Option)
# Part b

# General information and market-fitted parameters
T = 3  # Maturity time
T1 = 1.0729 # Option expiration time
F = 81 
X = 82
theta = 0.0262
r0 = 0.0381
kappa = 0.08169
mu = 0.015
C = 2.58
alpha = 0.02
beta = 0.413
sigma = 0.111

# Crank-Nicolson method for the bond pricing PDE
# Crank-Nicolson Scheme Parameters

iMax = 1000 # max number of time steps
jMax = 10000 # max number of space steps
kMax = 1000   # max number of PSOR relaxations
omega = 1.2  # over-relaxation parameter
tol = 1e-6 # tolerance for convergence

rMax = 5.0 # max interest rate
dr = rMax / jMax  # r step size
dt = T / iMax # time step size

# Numpy arrays for storing values
r = np.zeros(jMax+1)
t = np.zeros(iMax+1)
B_new = np.zeros(jMax+1)
B_old = np.zeros(jMax+1)

V_new = np.zeros(jMax+1)
V_old = np.zeros(jMax+1)

#V_old[j] = max(B_old[j] - X, 0.0)

for i in range(iMax+1):
    t[i] = i*dt

for j in range(jMax+1):
    r[j] = j*dr

# Record the value of the bond at maturity
B_old[:] = F

V_old[:] = np.maximum(B_old - X, 0.0)

# Matrix solution for the Crank-Nicolson scheme
# Storage for A (tridiagonal matrix)
A_bands = np.zeros(shape=(3,jMax+1))

band_structure = (1, 1) # (lower_bands, upper_bands)

# Allocate storage for the (bond) RHS term (d)
d = np.zeros(jMax+1) 

###

for i in range(iMax-1, -1, -1):

    # Fill in the tridiagonal matrix A
    A_bands.fill(0.0)

    # Special case for j = 0 (PDE boundary condition at r = 0)
    common_part_bc = (kappa * theta *
                        (np.exp(mu*i*dt) + np.exp(mu*(i+1)*dt))) \
                        / (2.0*dr)
    A_bands[1,0] = -1.0/dt - common_part_bc   # b_0
    A_bands[0,1] =  common_part_bc            # c_0

    # a[j], b[j], c[j] for matrix middle rows
    for j in range(1, jMax):
        j_term = (j**(2*beta)) * (dr**(2*beta - 2))
        k_part = kappa * (theta*(np.exp(mu*i*dt) + np.exp(mu*(i+1)*dt)) - j*dr)
        A_bands[2,j-1] = 0.25*sigma**2*j_term - k_part/(4*dr)  # a_j
        A_bands[1,j  ] = -1.0/dt - 0.5*sigma**2*j_term - 0.5*j*dr  # b_j
        A_bands[0,j+1] = 0.25*sigma**2*j_term + k_part/(4*dr)  # c_j

    # Boundary condition at for j = 0 jMax
    # a_jMax B_jMax-1 + b_jMax B_jMax = d_jMax
    A_bands[2,jMax-1] = 0.0   # a_jMax
    A_bands[1,jMax  ] = 1.0   # b_jMax

    # Fill in the RHS term (d) for the Crank-Nicolson scheme
    d.fill(0.0)
    #exp_term = 0.5*C*np.exp(-alpha*t[i]) + 0.5*C*np.exp(-alpha*t[i+1])
    exp_term = 0.5*C*np.exp(-alpha*(t[i]+0.5*dt))

    c0 = common_part_bc
    d[0] = (-1.0/dt + c0)*B_old[0] - c0*B_old[1] - exp_term

    for j in range(1, jMax):
        aa = A_bands[2,j-1]
        bb = A_bands[1,j] + 2.0/dt
        cc = A_bands[0,j+1]
        d[j] = -aa*B_old[j-1] - bb*B_old[j] - cc*B_old[j+1] - exp_term

    d[jMax] = 0.0   # d_jMax

    # Solve the equation
    B_new = solve_banded(band_structure, A_bands, d)
    B_old = np.copy(B_new)

    # Update the option value using the bond value
    
    if i == int(T1/dt):
        V_old = np.maximum(B_old - X, 0.0)   # Update V_old with the exercise values

    # excercise values
    E = np.maximum(B_old - X, 0.0)
    
    V_new = V_old.copy()
    
    a_opt = np.zeros(jMax+1)
    b_opt = np.zeros(jMax+1)
    c_opt = np.zeros(jMax+1)
    d_opt = np.zeros(jMax+1)
    
    # Special case for j = 0
    a_opt[0] = 0.0
    b_opt[0] = 1.0
    c_opt[0] = 0.0
    d_opt[0] = B_old[0] - X
    
    # a[j], b[j], c[j] for matrix middle rows
    for j in range(1, jMax):
        j_term = (j**(2*beta)) * (dr**(2*beta - 2))
        k_part = kappa * (theta*(np.exp(mu*i*dt) + np.exp(mu*(i+1)*dt)) - j*dr)
        
        a_opt[j] = 0.25*sigma**2*j_term - k_part/(4*dr)
        b_opt[j] = -1.0/dt - 0.5*sigma**2*j_term - 0.5*j*dr
        c_opt[j] = 0.25*sigma**2*j_term + k_part/(4*dr)
        d_opt[j] = (-a_opt[j]*V_old[j-1]+(-1.0/dt + 0.5*sigma**2*j_term + 0.5*j*dr)*V_old[j]-c_opt[j]*V_old[j+1])
        
    # Boundary condition at for j = jMax
    a_opt[jMax] = 0.0
    b_opt[jMax] = 1.0
    c_opt[jMax] = 0.0
    d_opt[jMax] = 0.0
    
    
    if i <= int(T1/dt):
        # Loop for PSOR method
        for k in range(kMax):
            epsilon = 0.0 # convergence parameter
            
            y = (1/b_opt[0]) * (d_opt[0] - c_opt[0]*V_new[1])
            y = V_new[0] + omega * (y - V_new[0])
            y = np.maximum(y, B_old[0] - X)
            
            epsilon += np.square(y - V_new[0])
            V_new[0] = y
            
            # Matrix middle rows
            for j in range(1, jMax):
                y = (1/b_opt[j]) * (d_opt[j] - a_opt[j]*V_new[j-1] - c_opt[j]*V_new[j+1])
                y = V_new[j] + omega * (y - V_new[j])
                y = np.maximum(y, B_old[j] - X)
                
                epsilon += np.square(y - V_new[j])
                V_new[j] = y

            # For j = jMax
            y = (1/b_opt[jMax]) * (d_opt[jMax] - a_opt[jMax]*V_new[jMax-1])
            y = V_new[jMax] + omega * (y - V_new[jMax])
            y = np.maximum(y, 0.0)
            
            epsilon += np.square(y - V_new[jMax])
            V_new[jMax] = y

            # Check for convergence
            if (epsilon < tol**2):
                break
            
            # Save values for plot
        if i == int(T1/dt):
            V_at_T1 = V_new.copy()
            B_at_T1 = B_new.copy()
    
        V_old = V_new.copy()

V_at_0 = V_old.copy()

tol = 1e-10
# Find maximum r for which the option is exercised
max_ex_index = np.maximum(B_at_T1 - X, 0.0) 
proximity_to_exercise = V_at_T1 - max_ex_index
j_indices_max_optimal = np.where((max_ex_index > 0) & (proximity_to_exercise < tol))[0].max() # to basically ignore irrelevant indices of i
r_indices_max_optimal = r[j_indices_max_optimal]

###
# Plot for V_at_T1 and V_at_0 against r
plt.figure(figsize=(10, 6))
plt.plot(r, V_at_T1, label=f'$V(r,\,t=T_1)$')
plt.plot(r, V_at_0,  label='$V(r,\,t=0)$')
plt.title('Value of American Call on Bond: $V$ vs $r$ at $t=T_1$ and $t=0$')
plt.xlabel('Interest Rate ($r$)')
plt.ylabel('American Call Value ($V$)')
plt.legend()
plt.grid()
#plt.xlim(0, rMax)
#plt.xlim(0, 0.1)
#plt.ylim(0, 100)
plt.show()


# Print the maximum r for which the option is exercised
print("Max r for which option is exercised:", r_indices_max_optimal)

# Printing the option values at r0
print("Option value at r0, t=0  :", V_at_0[int(r0/dr)])
print("Option value at r0, t=T1 :", V_at_T1[int(r0/dr)])

# Extra in 2.2 
# Function for Task 2.2 Extra
def Crank_PSOR_American(iMax, jMax, T, T1, F, X, theta, r0, kappa, mu, C, alpha, beta, sigma):
    # Numpy arrays for storing values
    r = np.zeros(jMax+1)
    t = np.zeros(iMax+1)
    B_new = np.zeros(jMax+1)
    B_old = np.zeros(jMax+1)

    V_new = np.zeros(jMax+1)
    V_old = np.zeros(jMax+1)

    #V_old[j] = max(B_old[j] - X, 0.0)

    for i in range(iMax+1):
        t[i] = i*dt

    for j in range(jMax+1):
        r[j] = j*dr

    # Record the value of the bond at maturity
    B_old[:] = F

    V_old[:] = np.maximum(B_old - X, 0.0)

    # Matrix solution for the Crank-Nicolson scheme
    # Storage for A (tridiagonal matrix)
    A_bands = np.zeros(shape=(3,jMax+1))

    band_structure = (1, 1) # (lower_bands, upper_bands)

    # Allocate storage for the (bond) RHS term (d)
    d = np.zeros(jMax+1) 

    ###

    for i in range(iMax-1, -1, -1):

        # Fill in the tridiagonal matrix A
        A_bands.fill(0.0)

        # Special case for j = 0 (PDE boundary condition at r = 0)
        common_part_bc = (kappa * theta *
                            (np.exp(mu*i*dt) + np.exp(mu*(i+1)*dt))) \
                            / (2.0*dr)
        A_bands[1,0] = -1.0/dt - common_part_bc   # b_0
        A_bands[0,1] =  common_part_bc            # c_0

        # a[j], b[j], c[j] for matrix middle rows
        for j in range(1, jMax):
            j_term = (j**(2*beta)) * (dr**(2*beta - 2))
            k_part = kappa * (theta*(np.exp(mu*i*dt) + np.exp(mu*(i+1)*dt)) - j*dr)
            A_bands[2,j-1] = 0.25*sigma**2*j_term - k_part/(4*dr)  # a_j
            A_bands[1,j  ] = -1.0/dt - 0.5*sigma**2*j_term - 0.5*j*dr  # b_j
            A_bands[0,j+1] = 0.25*sigma**2*j_term + k_part/(4*dr)  # c_j

        # Boundary condition at for j = 0 jMax
        # a_jMax B_jMax-1 + b_jMax B_jMax = d_jMax
        A_bands[2,jMax-1] = 0.0   # a_jMax
        A_bands[1,jMax  ] = 1.0   # b_jMax

        # Fill in the RHS term (d) for the Crank-Nicolson scheme
        d.fill(0.0)
        #exp_term = 0.5*C*np.exp(-alpha*t[i]) + 0.5*C*np.exp(-alpha*t[i+1])
        exp_term = 0.5*C*np.exp(-alpha*(t[i]+0.5*dt))

        c0 = common_part_bc
        d[0] = (-1.0/dt + c0)*B_old[0] - c0*B_old[1] - exp_term

        for j in range(1, jMax):
            aa = A_bands[2,j-1]
            bb = A_bands[1,j] + 2.0/dt
            cc = A_bands[0,j+1]
            d[j] = -aa*B_old[j-1] - bb*B_old[j] - cc*B_old[j+1] - exp_term

        d[jMax] = 0.0   # d_jMax

        # Solve the equation
        B_new = solve_banded(band_structure, A_bands, d)
        B_old = np.copy(B_new)

        # Update the option value using the bond value
        
        if i == int(T1/dt):
            V_old = np.maximum(B_old - X, 0.0)   # Update V_old with the exercise values

        # excercise values
        E = np.maximum(B_old - X, 0.0)
        
        V_new = V_old.copy()
        
        a_opt = np.zeros(jMax+1)
        b_opt = np.zeros(jMax+1)
        c_opt = np.zeros(jMax+1)
        d_opt = np.zeros(jMax+1)
        
        # Special case for j = 0
        a_opt[0] = 0.0
        b_opt[0] = 1.0
        c_opt[0] = 0.0
        d_opt[0] = B_old[0] - X
        
        # a[j], b[j], c[j] for matrix middle rows
        for j in range(1, jMax):
            j_term = (j**(2*beta)) * (dr**(2*beta - 2))
            k_part = kappa * (theta*(np.exp(mu*i*dt) + np.exp(mu*(i+1)*dt)) - j*dr)
            
            a_opt[j] = 0.25*sigma**2*j_term - k_part/(4*dr)
            b_opt[j] = -1.0/dt - 0.5*sigma**2*j_term - 0.5*j*dr
            c_opt[j] = 0.25*sigma**2*j_term + k_part/(4*dr)
            d_opt[j] = (-a_opt[j]*V_old[j-1]+(-1.0/dt + 0.5*sigma**2*j_term + 0.5*j*dr)*V_old[j]-c_opt[j]*V_old[j+1])
            
        # Boundary condition at for j = jMax
        a_opt[jMax] = 0.0
        b_opt[jMax] = 1.0
        c_opt[jMax] = 0.0
        d_opt[jMax] = 0.0
        
        
        if i <= int(T1/dt):
            # Loop for PSOR method
            for k in range(kMax):
                epsilon = 0.0 # convergence parameter
                
                y = (1/b_opt[0]) * (d_opt[0] - c_opt[0]*V_new[1])
                y = V_new[0] + omega * (y - V_new[0])
                y = np.maximum(y, B_old[0] - X)
                
                epsilon += np.square(y - V_new[0])
                V_new[0] = y
                
                # Matrix middle rows
                for j in range(1, jMax):
                    y = (1/b_opt[j]) * (d_opt[j] - a_opt[j]*V_new[j-1] - c_opt[j]*V_new[j+1])
                    y = V_new[j] + omega * (y - V_new[j])
                    y = np.maximum(y, B_old[j] - X)
                    
                    epsilon += np.square(y - V_new[j])
                    V_new[j] = y

                # For j = jMax
                y = (1/b_opt[jMax]) * (d_opt[jMax] - a_opt[jMax]*V_new[jMax-1])
                y = V_new[jMax] + omega * (y - V_new[jMax])
                y = np.maximum(y, 0.0)
                
                epsilon += np.square(y - V_new[jMax])
                V_new[jMax] = y

                # Check for convergence
                if (epsilon < tol**2):
                    break
                
                # Save values for plot
            if i == int(T1/dt):
                V_at_T1 = V_new.copy()
                B_at_T1 = B_new.copy()
        
            V_old = V_new.copy()

    V_at_0 = V_old.copy()

    return V_at_0

# Plotting the option value for varying jMax

rMax = 5.0
j_vals = [100, 200, 300, 400, 500, 1000, 5000, 10000]
V_list = []
plt.figure(figsize=(10, 6))

for jMax in j_vals:
    dr = rMax/jMax
    dt = T/iMax
    
    V_at_0_ = Crank_PSOR_American(iMax, jMax, T, T1, F, X, theta, r0, kappa, mu, C, alpha, beta, sigma)
    V_list.append(V_at_0_[int(r0/dr)])
    
plt.plot(j_vals, V_list, marker='o', linewidth=1, color='red')

plt.title('Option Value $V(r_0,0;T_1,T)$ Against Used $jMax$')
plt.xlabel('Max Number of Rate Iterations $jMax$')
plt.ylabel('Option Value')
#plt.legend()
plt.grid()
#plt.xlim(0, 5)
#plt.ylim(0, 20)
plt.show()