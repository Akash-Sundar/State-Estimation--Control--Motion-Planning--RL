import numpy as np
import matplotlib.pyplot as plt
from HMM_solution import HMM
import random


if __name__ == "__main__":

    # Load the data
    T = np.array([[0.5, 0.5], [0.5, 0.5]], dtype = float)
    M = np.array([[0.4, 0.1, 0.5], [0.1, 0.5, 0.4]])
    obs = np.array([2, 0, 0, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 0, 0, 1]).reshape(-1)
    pi = np.array([0.5, 0.5])

    #### Test your code here

    obj = HMM(obs, T, M, pi)

    alpha = obj.forward()
    print('alpha: \n', alpha)

    beta = obj.backward()
    print('beta: \n', beta)

    gamma = obj.gamma_comp(alpha, beta)
    print('gamma: \n', gamma)

    xi = obj.xi_comp(alpha, beta, gamma)
    print('xi: \n', xi)

    T_prime, M_prime, new_init_state = obj.update(alpha, beta, gamma, xi)
    print(new_init_state.shape)

    P_original, P_prime = obj.trajectory_probability(alpha, beta, T_prime, M_prime, new_init_state)
    print('P_original: \n', P_original)
    print('P_prime: \n', P_prime)

