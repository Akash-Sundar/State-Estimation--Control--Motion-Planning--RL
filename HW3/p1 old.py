# Policy Iteration

import numpy as np
import matplotlib.pyplot as plt

env_shape = (10,10)
N = env_shape[0] * env_shape[1]
gamma = 0.9 # discount factor

def create_occupancy_map():
    occ_map = np.zeros(env_shape)
    
    occ_map[:, 0] = 1
    occ_map[:,-1] = 1
    occ_map[0, :] = 1
    occ_map[-1,:] = 1

    occ_map[2, 3:6+1] = 1
    occ_map[4:7+1, 4] = 1
    occ_map[7, 5] = 1
    occ_map[4:5+1, 7] = 1

    occ_map = np.flip(occ_map)
    occ_map = np.flip(occ_map, axis = 1)

    return occ_map

def prob_transition(u_element, i, j):

    #u_element:
    #0 - North, 1 - East, 2 - West, 3 - South, 4 - no move

    T_element = np.zeros((3,3))
    idx = [1, 3, 4, 5, 7]
    np.put(T_element, idx, 0.1)

    if u_element == 0:
        T_element[0,1] = 0.7
        T_element[2,1] = 0

    elif u_element == 1:
        T_element[1,2] = 0.7
        T_element[1,0] = 0

    elif u_element == 2:
        T_element[1,0] = 0.7
        T_element[1,2] = 0

    elif u_element == 3:
        T_element[2,1] = 0.7
        T_element[0,1] = 0

    elif u_element == 4:
        # sticky obstacle
        T_element = np.zeros((3,3))
        T_element[1,1] = 1

    else:
        print("Invalid Control Input")
        return 0
    
    T_ij = T_element

    T_ij = np.zeros((env_shape[0]+2, env_shape[1]+2))
    i = i+1
    j = j+1

    T_ij[i-1:i+2, j-1:j+2] = T_element

    T_ij = T_ij[1:-1, 1:-1]
    
    return T_ij.flatten()

def generate_transition_matrix(u):

    # T = [prob_transition(u_element) for u_row in u for u_element in u_row]

    T = []

    for i in range(0, u.shape[0]):
        for j in range(0, u.shape[1]):
            T.append(prob_transition(u[i,j], i, j))

    T = np.array(T)
    # print(T.shape)
    assert T.shape == (N, N)

    return T

def plot_image(u):
    u = u.reshape(env_shape)

    u = np.flip(u, axis = 1)
    u = np.flip(u)
    
    plt.figure()
    plt.imshow(u, cmap='jet', origin='lower')
    plt.colorbar()
    plt.title('Value Function for Initial Policy')
    plt.show()

def policy_evaluation(T, Q, J_init, u, gamma):

    iter = 300
    J = np.zeros((iter, 10, 10))
    J[0] = J_init

    for k in range(1, iter):
        for i in range(10):
            for j in range(10):
                J[k,i,j] = Q[i,j,u[i,j]] + gamma*(T[i,j,u[i,j],:,:].reshape((1,-1))@J[k-1,:,:].reshape((-1,1)))

    return J[-1]

def policy_improvement(Q, gamma, J_pi):

    epsilon = np.random.randn(N)
    E = Q + gamma * J_pi + epsilon.reshape((N,1))

    # u_next = np.argmin(E, axis = 1)
    u_next = E
    # Pi_next = u_next

    return u_next #, Pi_next

occ_map = create_occupancy_map()    
# print(occ_map)

u = np.ones(env_shape)
u[occ_map == 1] = 4
# print(u.shape)

T = generate_transition_matrix(u)
# print(T)

Q = -1 * np.ones(env_shape) #reward 
Q[occ_map == 1] = -10
Q[8,1] = 10 #goal node

Pi = np.zeros(env_shape)
Pi = u

u = u.reshape(N, 1)
Q = Q.reshape(N, 1)
# print(u.shape)

u_old = np.empty_like(u)
plot_image(u)

epoch = 1

while not(np.array_equal(u_old, u)) and epoch < 5:

    u_old = u
    epoch = epoch + 1
    J_init = np.zeros(( 10, 10))
    J_pi = policy_evaluation(T, Q, J_init, u, gamma)
    u = policy_improvement(Q, gamma, J_pi)

    print(u.reshape(env_shape))

    plot_image(u)








