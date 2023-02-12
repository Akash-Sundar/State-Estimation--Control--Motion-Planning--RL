import numpy as np
import matplotlib.pyplot as plt
from histogram_filter import HistogramFilter
import random


if __name__ == "__main__":

    # Load the data
    data = np.load(open('starter.npz', 'rb'))
    cmap = data['arr_0']
    actions = data['arr_1']
    observations = data['arr_2']
    belief_states = data['arr_3']

    # print("belief_states: \n", belief_states)
    # print(belief_states.shape)

    # print("cmap: \n", cmap)
    # print(cmap.shape)

    # print("actions: \n", actions)
    # print(actions.shape)

    # print("observations: \n", observations)
    # print(observations.shape)


    #### Test your code here

    bel_init = 1/400 * np.ones([20,20])
    
    obj1 = HistogramFilter()
    for i in range(0,30):
        u = actions[i]
        z = observations[i]
        bel_init,ind = obj1.histogram_filter(cmap, bel_init, u , z)
        

    print('##')
    print(bel_init.shape)
    print(ind)
