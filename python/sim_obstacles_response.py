# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 19:15:51 2023

@author: dansh
"""

import numpy as np
import multiprocessing

def sim(F, n):
    # Setting the constants
    # F = 0.05 # Force 
    L = 1000 # lattice size
    # n = 0.01 # obstacle density 
    
    N = int(1e4) # length of trajectory
    M = int(1e4) # amount of trajectories
    
    N_obs = int(n * L * L)
    Dy = 1/(np.exp(F/2)+np.exp(-F/2)+2) # normilization of transition probabilties 
    rng=np.random.default_rng() 
    p_right = Dy * np.exp(F/2)
    p_left = Dy * np.exp(-F/2)
    p_up = Dy
    p_down = Dy
    
    # Single obstacle realization
    # we run many long trajectories with periodic boundary conditions.
    # So as to sample all the space.
    # We look at the distance traveled on the x axis for each k steps up to a 
    # max of k_max = N to calculate the average position.
    
    # Setting the obstacles position once for many trajectories 
    ###############################
    # draw the obstacles
    # Randomize the location of the obstacles
    s_mat = np.zeros([L, L], np.int32)
    
    for i in range(N_obs):
        bool_repeat = True
        while bool_repeat:
            sx = rng.choice(L)
            sy = rng.choice(L)
            if s_mat[sx,sy] == 0:
                s_mat[sx,sy] = 1
                bool_repeat = False
    
    ################################
    steps=rng.choice([[1,0],[-1,0],[0,1],[0,-1]], [N, M], p=[p_right, p_left, p_up, p_down])
    x_avg = np.zeros(N+1)
    x2_avg = np.zeros(N+1)
    
    for traj_id in range(M):
        x = np.zeros(N+1, dtype = np.dtype(np.int32))
        y = np.zeros(N+1, dtype = np.dtype(np.int32))
        
        # choose where to start
        bool_repeat = True
        while bool_repeat:
            x0 = rng.choice(L)
            y0 = rng.choice(L)
            if s_mat[x0,y0] == 0:
                bool_repeat = False
        x[0] = x0
        y[0] = y0
        
        x_steps = steps[:,traj_id,0]
        y_steps = steps[:,traj_id,1]
        
        for i in range(N):
            
            x[i+1] = x[i] + x_steps[i]
            y[i+1] = y[i] + y_steps[i]
            
            x_periodic = x[i+1] - int(x[i+1]/L) * L
            y_periodic = y[i+1] - int(y[i+1]/L) * L
                
            # the move is rejected if we end up at an obstacle site
            if s_mat[x_periodic, y_periodic] == 1:
                x[i+1] = x[i]
                y[i+1] = y[i]
        x_avg = x_avg + (x - x0) / M
        x2_avg = x2_avg + (x - x0) ** 2 / M
        
    return x_avg, x2_avg
    
        
def wrap_sim(run_num):
    
    F = np.array([0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9])
    n = 0.05 # obstacle density 
    
    for itr in range(F.size):
        x_avg, x2_avg = sim(F[itr], n) 
        file_str='data_x2_' + 'F_' + str(F[itr]) + '_n_' + str(n) + '_run_' + str(run_num) + '.npy'
        with open(file_str, 'wb') as f:
            np.save(f, x2_avg)
    
def main():
    
    processes_num=60
    iterable=range(0,300);
    pool = multiprocessing.Pool(processes=processes_num)
#    func = partial(sim, N_samples)
    pool.map(wrap_sim, iterable)
    pool.close()
    pool.join()
    print('Finished.')
    
if __name__ == "__main__":
    main()
