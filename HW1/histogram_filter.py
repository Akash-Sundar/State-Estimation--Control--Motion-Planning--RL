import numpy as np


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        '''

        ### Your Algorithm goes Below.
        cmap = np.array(cmap)
        action = np.array(action)
        n,m = cmap.shape
        
        state = 0.9*np.eye(m-1) 
        state_next = np.append(np.zeros([m-1,1]),state,axis=1)
        state_next = np.append(state_next,np.zeros([1,m]),axis=0)

        M = state_next+0.1*np.eye(m)
        M[-1,-1]=1

        # print('M: ', M)
        # print(M.shape)


        if action[0]==1:        
            alpha = np.matmul(belief, M)
        elif action[0]==-1:    
            alpha = np.flip(belief,1)
            alpha = np.flip(np.matmul(alpha, M),1)
        elif action[1]==1:    
            alpha = np.flip(belief.T,1)
            alpha = np.flip(np.matmul(alpha, M), 1).T
        else:                   
            alpha = belief.T
            alpha = np.matmul(alpha, M).T

        # print('alpha', alpha)
        # print(alpha.shape)


        if (observation ==1):            
            idx = np.where(cmap==0)
            Transition = 0.9*np.ones([n,m])
            Transition[idx]=0.1
            
        elif(observation ==0):
            idx = np.where(cmap==0)
            Transition = 0.1*np.ones([n,m])
            Transition[idx]=0.9

        # print('Transition', Transition)
        # print(Transition.shape)
        
        alpha = Transition*alpha

        alpha_normalized = np.array(alpha/np.sum(alpha))

        # print('alpha_normalized', alpha_normalized)
        # print(alpha_normalized.shape)
        
        return alpha_normalized