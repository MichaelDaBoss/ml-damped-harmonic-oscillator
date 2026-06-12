import numpy as np

# analytical solution to the underdamped HO
'''Assume the mass to be m = 1, and use initial conditions x(0) = 1 and amplitude A = 1'''
def analytic_sol(t, kappa, beta):
    omega = np.sqrt(kappa)*np.sqrt(1-beta**2/(4*kappa))
    return np.exp(-beta*t/2)*np.cos(omega*t)

# define ranges for the damping coefficient and spring constant
beta_range = np.linspace(0.5, 1.0, 100) # no ppoint in beta and kappa ranges as they are not directly used to pick out a random integer later, they are only used to define lower and upper bounds which can be done without these lists
kappa_range = np.linspace(5, 10, 100)
time_range = np.linspace(0, 10, 100)

# initialize empty arrays for x and y
X_array = np.zeros((5000, 100))
Y_array = np.zeros((5000, 2))

for i in range(5000):
    rand_k = np.random.uniform(np.min(kappa_range), np.max(kappa_range))
    rand_b = np.random.uniform(np.min(beta_range), np.max(beta_range))

    X_array[i,:] = analytic_sol(time_range, rand_k, rand_b)

    Y_array[i,:] = (rand_k, rand_b)

# generate noise in the data using normal Gaussian noise with a small standard deviation and mean of 0
noisy_X_array = X_array + np.random.normal(0, 0.05, size=X_array.shape)

# save arrays to compressed .npz file
np.savez_compressed('datasets/data.npz', X=noisy_X_array, y=Y_array, time=time_range)