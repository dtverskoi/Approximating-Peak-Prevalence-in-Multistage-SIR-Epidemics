import numpy as np
from functions import *


if __name__ == '__main__':
    # Initialize vectors with r0 and k values
    r0_vect = np.arange(1.5, 10.5, 0.5)
    k_vect = np.arange(1, 21, 1)# [0, 1, 2, ... 9]
    t_max = 200
    epsilon = 0.01
    gamma = 0.1

    # Call wrapper function to return estimates
    bias, estimates = wrapper(r0_vect, k_vect, t_max, epsilon, gamma)

    # Plot results of bias
    plot_bias_results(bias)