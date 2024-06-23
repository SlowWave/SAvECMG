import os
import sys

# add parent directory to "sys.path" to import modules from that path
sys.path.append(os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir))

from modules.sc_body import SpacecraftBody
import plotly.graph_objects as go
import numpy as np

# set S/C inertia and initial quaternion and rate
quaternion = [1, 0, 0, 0]
rate = [0, 0, 0]
inertia = np.eye(3)

# initialize S/C object
sc_body = SpacecraftBody(quaternion, rate, inertia)

# initialize lists for quaternion and rate
quaternion = list()
rate = list()

# set simulation timestep
time_step = 0.1

# create timespan
timespan = np.linspace(0, 1000, 1000) * time_step

# simulate S/C dynamics
for i in range(1000):
    sc_body.propagate_states(
        external_torque=np.array([0.01, 0, 0]),
        cmga_torque=np.array([0, 0, 0]),
        cmga_angular_momentum=np.array([0, 0, 0]),
        time_step=time_step,
    )
    states = sc_body.get_states()
    quaternion.append(states[0])
    rate.append(states[1])

# plot results
fig = go.Figure()

fig.add_trace(go.Scatter(x=timespan, y=[row[0] for row in quaternion]))
fig.add_trace(go.Scatter(x=timespan, y=[row[1] for row in quaternion]))
fig.add_trace(go.Scatter(x=timespan, y=[row[2] for row in quaternion]))
fig.add_trace(go.Scatter(x=timespan, y=[row[3] for row in quaternion]))

fig.show()

fig = go.Figure()

fig.add_trace(go.Scatter(x=timespan, y=[row[0] for row in rate]))
fig.add_trace(go.Scatter(x=timespan, y=[row[1] for row in rate]))
fig.add_trace(go.Scatter(x=timespan, y=[row[2] for row in rate]))

fig.show()
