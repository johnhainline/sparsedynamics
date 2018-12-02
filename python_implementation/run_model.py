import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from scipy.signal import gausspulse
from scipy.integrate import odeint
from mouse_model import OrganModel, mouse
from utility_functions import sparsifyDynamics, poolData



def gauss_env(x,mu=.8,sig=.1):
	return np.exp(-.5*(x-mu)**2/(sig**2))

def plot_traj(name):
	ix = mouse.get_var(name).ix
	traj = trajectory[:,ix]
	plt.plot(t_span,traj)
	plt.title(name)
	plt.show()

def plot_ss(name1,name2):
	ixs = mouse.get_var(name1).ix, mouse.get_var(name2).ix
	trajs = trajectory[:,ixs]
	plt.plot(trajs[0,:],trajs[1,:])
	plt.title('{} vs {}'.format(name1,name2))
	plt.show()



n_states = len(mouse.state_vars)

# impulse input to arterial blood
arterial_blood_ix = mouse.get_var('Arterial Blood').ix

impulse_mag = 100
input_fn = lambda t: np.array([gauss_env(t)*int(ix==arterial_blood_ix) for ix in range(n_states)])
mouse.input_fn = input_fn


# # test eval_derivative fn and input fn
# rstate = lambda : np.random.normal(size=n_states)
# zstate = np.zeros(n_states)
# d_dt = mouse.eval_derivative(at_state=zstate, t=0)
# print(d_dt)

# intial state
x0 = np.zeros(n_states)
t_start = 0
t_end = 40
t_step = .05
t_span = np.arange(t_start,t_end,t_step)

# odeint(dynamic func, initial point, time points to evaluate
trajectory = odeint(mouse.eval_derivative,x0,t_span)


# drop t before impulse is gone
t_impulse_gone = 1
ix_impulse_gone = int((t_impulse_gone-t_start)/t_step)



cols = ["({})".format(sv.name) for sv in mouse.state_vars]
X = pd.DataFrame(data=trajectory, columns=cols)
X = X[ix_impulse_gone:]

# noise = 1*np.random.normal(size=X.shape)
# X = X+noise

dX = X.diff().dropna().reset_index(drop=True)
tX = X[:-1].reset_index(drop=True)

tX.plot()
plt.show()

print(X.max(),X.min())
print(dX.max(),dX.min())


# SINDy params
nVars = len(cols)
polyorder = 1
lam = 0.00001
usesine = 0

# generate data
Theta = poolData(tX,nVars, polyorder, usesine)
dot_labels = pd.Index([s + ' [dot]' for s in cols])

# run SINDy
dXh = sparsifyDynamics(Theta,dX,lam)
dXh = dXh.set_index(dot_labels)
print(dXh.head())




# plot_traj('Arterial Blood')


# # plot some trajectories

# var_names = [sv.name for sv in mouse.state_vars]

# pairs = itertools.combinations(var_names,2)
# pairs = list(pairs)

# while True:
# 	pair = pairs[np.random.choice(len(pairs))]
# 	v1,v2 = pair
# 	plot_traj(v1)


