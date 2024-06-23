import os
import sys

# add parent directory to "sys.path" to import modules from that path
sys.path.append(os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir))

from modules.cmga import ControlMomentGyro
import plotly.graph_objects as go
import numpy as np

# initialize CMG object
cmg = ControlMomentGyro(theta=0, theta_dot=0, theta_dot_max=0.1, angular_momentum=10)

# create lists for theta and theta_dot
theta = list()
theta_dot = list()

# set simulation timestep and theta_dot reference
time_step = 0.1
theta_dot_ref = 0.3

# create timespan
timespan = np.linspace(0, 1000, 1000) * time_step

# simulate CMG dynamics
for i in range(1000):
    cmg.propagate_states(theta_dot_ref=theta_dot_ref, time_step=time_step)
    states = cmg.get_states()
    theta.append(states[0])
    theta_dot.append(states[1])

# plot results
fig_1 = go.Figure()
fig_1.add_trace(go.Scatter(x=timespan, y=theta))
fig_1.show()

fig_2 = go.Figure()
fig_2.add_trace(go.Scatter(x=timespan, y=theta_dot))
fig_2.show()
