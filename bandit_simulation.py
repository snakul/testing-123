import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def display_setting():
    pd.set_option('display.max_rows',100, 'display.max_columns',50, 'display.width',1000, 'display.precision',2, 'display.float_format', lambda x: '%.2f' %x )
    np.set_printoptions(precision = 2, linewidth = 1000, threshold = 5000, suppress = True)

display_setting()

np.random.seed(0)
N_actions = 10
N_repeat = 10000
N_experiments = 2000
R_experiments = np.arange(N_experiments)

L_epsilons = [0.1, 0.05, 0]
Qa = np.random.randn(N_actions, N_experiments) # true value of expected reward # kept same for each epsilon to compare fairly
noise = 5*np.random.randn(N_repeat, N_experiments)
plt.figure()
colors = ['mediumpurple','indianred','steelblue']
for ei, epsilon in enumerate (L_epsilons):
    T_Qa = np.zeros((N_actions, N_experiments)) # total reward at any point in time for each exp for each state
    C_Qa = np.zeros((N_actions, N_experiments)) # count at any point (depends how many times a particular choice of action was made)
    E_Qa = np.ones((N_actions, N_experiments)) # T_Qa/C_Qa # estimated value of expected reward- separate for each exp # starts at high value so all actions are selected
    Rt_df = pd.DataFrame(np.zeros((N_repeat, N_experiments)))

    for t in np.arange(N_repeat):
        if t == 0:
            a = np.random.randint(0, N_actions, N_experiments)
        else:
            a = E_Qa.argmax(0) #greedy action # indices of max estimated reward for each exp
            throws = np.random.uniform (0,1, N_experiments)
            a[throws< epsilon] = np.random.randint (0, N_actions, (throws<epsilon).sum())
        Rt_df.ix[t,:] = Qa [a, R_experiments] + noise[t,:] # actual reward received = true expected + noise
        T_Qa[a, R_experiments] = T_Qa  [a, R_experiments] + Rt_df.ix[t,:]
        C_Qa[a, R_experiments] = C_Qa[a, R_experiments] + 1
        E_Qa[a, R_experiments] = T_Qa[a, R_experiments] /  C_Qa[a, R_experiments]

    plt.plot( Rt_df.mean(axis = 1), label = epsilon, color = colors[ei] )
    plt.legend( loc = 'best' )

plt.show()

