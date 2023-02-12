import numpy as np

class HMM():

    def __init__(self, Observations, Transition, Emission, Initial_distribution):
        self.Observations = Observations
        self.Transition = Transition
        self.Emission = Emission
        self.Initial_distribution = Initial_distribution

    def forward(self):

        M = self.Emission
        T = self.Transition
        pi = self.Initial_distribution

        n = len(self.Observations)
        alpha = np.zeros((self.Observations.shape[0], self.Transition.shape[0]))
        alpha[0] = pi * M[:,self.Observations[0]]

        for k in range(0, n-1):
            alpha[k+1] = np.multiply(M[:,self.Observations[k+1]].T, alpha[k].T @ T).T        

        return alpha


    def backward(self):

        M = self.Emission
        T = self.Transition

        n = self.Observations.shape[0]

        beta = np.zeros((self.Observations.shape[0], self.Transition.shape[0]))
        beta[-1] = np.ones(self.Transition.shape[0])

        for k in range(n-2, -1, -1):
            beta[k] = np.multiply(beta[k+1], M[:,self.Observations[k+1]]) @ T

        return beta

    def gamma_comp(self, alpha, beta):

        alpha_sum = np.sum(alpha[-1], axis = 0)
        
        gamma = np.zeros((self.Observations.shape[0], self.Transition.shape[0]))

        n = len(self.Observations)
        for k in range(0, n):
            gamma[k] = np.exp(np.log(alpha[k]) + np.log(beta[k]))

        gamma = gamma/alpha_sum

        return gamma

    def xi_comp(self, alpha, beta, gamma):

        T = self.Transition
        xi = np.zeros(( self.Observations.shape[0] - 1, T.shape[0], T.shape[0] ))
        for k in range(self.Observations.shape[0] - 1):
            xi[k] = np.dot( np.dot( np.diag( alpha[k] ), T ), np.diag( beta[k + 1] * self.Emission[ :, self.Observations[k + 1] ].T ) ) / np.sum( np.dot( np.dot( np.diag( alpha[k] ), T ), np.diag( beta[k + 1] * self.Emission[ :, self.Observations[k+1]].T)))
        return xi

    def update(self, alpha, beta, gamma, xi):

        M_prime = np.zeros_like(self.Emission)
        T_prime = np.zeros_like(self.Transition)

        new_init_state = gamma[0]

        for i in range(T_prime.shape[0]):
            for j in range(T_prime.shape[0]):
                for k in range(0, self.Observations.shape[0]-1):
                    T_prime[i,j] = T_prime[i,j] + xi[k,i,j]
                gamma_sum = np.sum(gamma[:-1, i])
                T_prime[i,j] = T_prime[i,j]/ gamma_sum

        for i in range(M_prime.shape[1]):
            for j in range(T_prime.shape[0]):
                for k in range(0, self.Observations.shape[0]):
                    if (self.Observations[k] == i):
                        M_prime[j,i] = M_prime[j,i] + gamma[k,j]                   
                gamma_sum = np.sum(gamma[:, j])
                M_prime[j,i] = M_prime[j,i]/ gamma_sum
    
        return T_prime, M_prime, new_init_state

    def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state):

        T = self.Transition

        P_original = np.array([0.])
        P_prime = np.array([0.])

        alpha_prime = np.zeros(( self.Observations.shape[0], T.shape[0] ))
        alpha_prime[0] = new_init_state * M_prime[:, self.Observations[0]]
        for k in range(self.Observations.shape[0] - 1):
            alpha_prime[k + 1] = np.dot( alpha_prime[k], T_prime ) * M_prime[:, self.Observations[k + 1]]

        for i in range( T.shape[0] ):
            P_original= P_original + alpha[self.Observations.shape[0]-1][i]
            P_prime= P_prime + alpha_prime[self.Observations.shape[0]-1][i]

        return P_original, P_prime
