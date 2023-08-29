import gym
import numpy as np
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions.uniform import Uniform as uniform_distribution
import copy

q_value = lambda  q, x, u: q(x)[range(len(u)), u.long()]

def rollout(e, q, eps=0, T=200):
    traj = []

    x = e.reset()[0]
    for t in range(T):
        u = q.control(th.from_numpy(x).float().unsqueeze(0),
                      eps=eps)
        u = u.int().numpy().squeeze()

        xp,r,d,_, info = e.step(u)
        t = dict(x=x,xp=xp,r=r,u=u,d=d,info=info)
        x = xp
        traj.append(t)
        if d:
            break
    return traj

class q_t(nn.Module):
    def __init__(s, xdim, udim, hdim=16):
        super().__init__()
        s.xdim, s.udim = xdim, udim
        s.m = nn.Sequential(
                            nn.Linear(xdim, hdim),
                            nn.ReLU(True),
                            nn.Linear(hdim, udim),
                            )
    def forward(s, x):
        return s.m(x)

    def control(s, x, eps=0):
        # 1. get q values for all controls
        q = s.m(x)

        ### TODO: XXXXXXXXXXXX
        # eps-greedy strategy to choose control input
        # note that for eps=0
        # you should return the correct control u

        if np.random.rand() <= eps:
            low = th.tensor([0.0])
            high = th.tensor([s.udim])
            u = uniform_distribution(low, high).sample((1,1))
        else:
            u = th.argmax(q)

        return u
    
def sample_mini_batch(ds, batch_size, gamma):

    batch = np.squeeze(np.random.choice(ds, batch_size))

    x_batch = th.cat([th.stack([th.from_numpy(p['x']) for p in traj]) for traj in batch])
    u_batch = th.cat([th.stack([th.from_numpy(p['u']) for p in traj]) for traj in batch])
    xp_batch = th.cat([th.stack([th.from_numpy(p['xp']) for p in traj]) for traj in batch])
    r_batch = th.cat([th.tensor([p['r']*gamma**k for k, p in enumerate(traj)]) for traj in batch])
    d_batch = th.cat([th.tensor([int(p['d']) for p in traj]) for traj in batch])

    traj = x_batch, u_batch, xp_batch, r_batch, d_batch

    return traj

    

def loss(q, q_target, ds):
    ### TODO: XXXXXXXXXXXX
    # 1. sample mini-batch from datset ds
    # 2. code up dqn with double-q trick
    # 3. return the objective f

    # u* = argmax q(x', u)
    # (q(x, u) - r - g*(1-indicator of terminal)*qc(x' , u*))**2

    batch_size = 16
    gamma = 0.9
    
    x, u, xp, r, d = sample_mini_batch(ds, batch_size, gamma)

    u_star = th.argmax(q(xp), dim=1)

    f = th.mean(th.square(q_value(q, x, u) - gamma*(1 - d)*q_value(q_target, xp, u_star) - r))
    return f


def evaluate(q):
    ### TODO: XXXXXXXXXXXX
    # 1. create a new environment e
    # 2. run the learnt q network for 100 trajectories on
    # this new environment to take control actions. Remember that
    # you should not perform epsilon-greedy exploration in the evaluation
    # phase
    # and report the average discounted
    # return of these 100 trajectories

    e = gym.make('CartPole-v1')

    gamma = 0.9
    num_trajectories = 100

    with th.no_grad():
        total_rewards = 0
        for _ in range(num_trajectories):
            traj = rollout(e, q, eps=0)
            traj_rewards = np.sum([p['r'] * gamma ** k for k, p in enumerate(traj)])
            total_rewards += traj_rewards
        r = total_rewards / num_trajectories
    return r

def visualize(fig_num, fig_name, data):
    plt.figure(fig_num) 
    plt.clf()
    plt.title('Average {} reward'.format(fig_name))
    plt.plot(np.arange(0, 1000*len(data), 1000), data, label=fig_name)
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Average reward')
    plt.savefig('p2_2_{}.png'.format(fig_name), format='png')
    plt.show()


if __name__=='__main__':

    e = gym.make('CartPole-v1')

    xdim, udim =    e.observation_space.shape[0], \
                    e.action_space.n

    q = q_t(xdim, udim, 8)
    q_target = copy.deepcopy(q)

    gamma = 0.9
    alpha = 0.05
    # Adam is a variant of SGD and essentially works in the
    # same way
    optim = th.optim.Adam(q.parameters(), lr=1e-3,
                          weight_decay=1e-4)

    ds = []
    for i in range(1000):
        ds.append(rollout(e, q, eps=1, T=200))


    rewards = []
    train_rewards = []
    for i in range(20000):
        q.train()
        t = rollout(e, q)
        ds.append(t)

        # perform weights updates on the q network
        # need to call zero grad on q function
        # to clear the gradient buffer
        q.zero_grad()
        f = loss(q, q_target, ds)
        f.backward()
        optim.step()

        for p, p_target in zip(q.parameters(), q_target.parameters()):
            p_target.data = (1 - alpha) * p_target.data + alpha * p.data

        if i%1000 == 0:    
            reward = evaluate(q)        
            rewards.append(reward)

            with th.no_grad():                
                trajectories = [rollout(e, q, eps=0) for _ in range(10)]
                train_reward = np.mean([np.sum([x['r']*gamma**k for k, x in enumerate(traj)]) for traj in trajectories])
                train_rewards.append(train_reward)            

            print('Average return for iteration %d is:'%(i), reward)           

    visualize(0, 'train', train_rewards)
    visualize(1, 'eval', rewards)

    exit()