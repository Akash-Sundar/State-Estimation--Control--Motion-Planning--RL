import numpy as np
from matplotlib import pyplot

env_shape = (10, 10)
N = env_shape[0] * env_shape[1]

reward = 10
gamma = 0.9 

# def prob_transition(u_element, i, j):

#     #u_element:
#     #0 - North, 1 - East, 2 - West, 3 - South, 4 - no move

#     T_element = np.zeros((3,3))
#     idx = [1, 3, 4, 5, 7]
#     np.put(T_element, idx, 0.1)

#     if u_element == 0:
#         T_element[0,1] = 0.7
#         T_element[2,1] = 0

#     elif u_element == 1:
#         T_element[1,2] = 0.7
#         T_element[1,0] = 0

#     elif u_element == 2:
#         T_element[1,0] = 0.7
#         T_element[1,2] = 0

#     elif u_element == 3:
#         T_element[2,1] = 0.7
#         T_element[0,1] = 0

#     elif u_element == 4:
#         # sticky obstacle
#         T_element = np.zeros((3,3))
#         T_element[1,1] = 1

#     else:
#         print("Invalid Control Input")
#         return 0
    
#     T_ij = T_element

#     T_ij = np.zeros((env_shape[0]+2, env_shape[1]+2))
#     i = i+1
#     j = j+1

#     T_ij[i-1:i+2, j-1:j+2] = T_element

#     T_ij = T_ij[1:-1, 1:-1]
    
#     return T_ij.flatten()

# def generate_transition_matrix(u):

#     # T = [prob_transition(u_element) for u_row in u for u_element in u_row]

#     T = []

#     for i in range(0, u.shape[1]):
#         for j in range(0, u.shape[2]):
#             T_k = []
#             for k in range(0, u.shape[0]):
#                 T_k.append(prob_transition(u[k,i,j], i, j))
#             T.append(T_k)

#     T = np.array(T)
#     print(T.shape)    

#     return T

def generate_T(obstacles_list, goal_idx):
    # 0 - left
    # 1 - right
    # 2 - up
    # 3 - down

    n = env_shape[0]
    T = np.zeros((n, n, 4, n, n))
    goal_row, goal_col = goal_idx
    for i in range(n):
        for j in range(n):
            if (i,j) == goal_idx:
                T[goal_row, goal_col, :, goal_row, goal_col] = 1
            elif (i,j) not in obstacles_list:
                T[i, j, 0, i, np.clip(j-1, 0, n-1)] += 0.7
                T[i, j, 0, np.clip(i-1, 0, n-1), j] += 0.1
                T[i, j, 0, np.clip(i+1, 0, n-1), j] += 0.1
                T[i, j, 0, i, j] += 0.1
                
                T[i, j, 1, i, np.clip(j+1, 0, n-1)] += 0.7
                T[i, j, 1, np.clip(i-1, 0, n-1), j] += 0.1
                T[i, j, 1, np.clip(i+1, 0, n-1), j] += 0.1
                T[i, j, 1, i, j] += 0.1
                
                T[i, j, 2, np.clip(i-1, 0, n-1), j] += 0.7
                T[i, j, 2, i, np.clip(j+1, 0, n-1)] += 0.1
                T[i, j, 2, i, np.clip(j-1, 0, n-1)] += 0.1
                T[i, j, 2, i, j] += 0.1
                
                T[i, j, 3, np.clip(i+1, 0, n-1), j] += 0.7
                T[i, j, 3, i, np.clip(j+1, 0, n-1)] += 0.1
                T[i, j, 3, i, np.clip(j-1, 0, n-1)] += 0.1
                T[i, j, 3, i, j] += 0.1

    return T

def generate_state_map():
    state_map = np.zeros(env_shape)
    
    state_map[:, 0] = 1
    state_map[:,-1] = 1
    state_map[0, :] = 1
    state_map[-1,:] = 1

    state_map[2, 3:6+1] = 1
    state_map[4:7+1, 4] = 1
    state_map[7, 5] = 1
    state_map[4:5+1, 7] = 1

    state_map = np.flip(state_map, axis = 1)
    state_map = np.flip(state_map)   

    state_map[8,8] = 2
    state_map[8,1] = 3
    state_map[3,3] = 4 

    return state_map

def generate_Q_map(states):
    obstacles = np.where(states == 1)
    goal_idx = np.where(states == 2)

    Q = -1*np.ones((env_shape[1], env_shape[0], 4)) 
    Q[obstacles[0], obstacles[1], :] = -reward
    Q[goal_idx[0], goal_idx[1], :] = reward

    return Q

def visualize(J_new, u):

    print("Control policy: \n", u)
    fig = pyplot.figure(figsize=(5,5))
    ax = fig.gca()
    ax.set_xticks(np.arange(0.5, 10.5, 1))
    ax.set_yticks(np.arange(0.5, 10.5, 1))
    pyplot.imshow(J_new)
    pyplot.grid()

    for i in range(10):
        for j in range(10):
            if (i,j) not in obstacles_list and (i,j) not in goal_idx_list:
                if u[i,j] == 0:
                    pyplot.arrow(j, i, -0.25, 0, head_width=0.1)
                elif u[i,j] == 1:
                    pyplot.arrow(j, i, 0.25, 0, head_width=0.1)
                elif u[i,j] == 2:
                    pyplot.arrow(j, i, 0, -0.25, head_width=0.1)
                elif u[i,j] == 3:
                    pyplot.arrow(j, i, 0, 0.25, head_width=0.1)
    pyplot.show()

def policy_evaluation(T, Q, J_init, u):
    iter = 300
    J = np.zeros((iter, env_shape[0], env_shape[1]))
    J[0] = J_init
    for k in range(1, iter):
        for i in range(10):
            for j in range(10):
                J[k,i,j] = Q[i,j,u[i,j]] + gamma*(T[i,j,u[i,j]].flatten() @ J[k-1].flatten().T)

    return J[-1]

def policy_improvement(T, Q, J_new):

    E = lambda x: np.argmax(x)

    u_k = np.zeros(env_shape)

    for i in range(env_shape[0]):
        for j in range(env_shape[1]):
            Q_element = Q[i,j]
            T_element = T[i,j].reshape(4, -1)
            
            u_k[i, j] = E(Q_element + (gamma* T_element @ J_new.reshape((-1,1)) ).reshape(4))

    return u_k


states = generate_state_map()

# 0, 1, 2, 3 # left, right, up, down
obstacles = np.where(states == 1)
obstacles_list = list(zip(*np.where(states == 1)))
goal_idx_list = list(zip(*np.where(states == 2)))

T = generate_T(obstacles_list, goal_idx_list[0])

Q = generate_Q_map(states)

num_policy_iter = 4
J = np.zeros(env_shape)
u = np.ones((num_policy_iter+1, env_shape[0], env_shape[1]), dtype=int)

for k in range(num_policy_iter):
    J_new = policy_evaluation(T, Q, J, u[k])
    J = J_new
    u[k+1] = policy_improvement(T, Q, J_new)
    visualize(J_new, u[k+1])
















