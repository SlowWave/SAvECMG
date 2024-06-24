import os
import sys

# add parent directory to "sys.path" to import modules from that path
sys.path.append(os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir))

from modules.sc_body import SpacecraftBody
from modules.aoc import AttitudeController
import plotly.graph_objects as go
import numpy as np


# set S/C inertia and initial quaternion and rate
quaternion = [1, -0.5, 0.5, -1]
rate = [0.1, 0, -0.3]
inertia = np.eye(3)

# initialize S/C object
sc_body = SpacecraftBody(quaternion, rate, inertia)

# initialize AOC object
aoc = AttitudeController(1, 1)

# set reference quaternion and rate
reference_quaternion = [0, 0, 0, 1]
reference_rate = [0, 0, 0]
aoc.set_reference(reference_quaternion, reference_rate)


# initialize lists for quaternion and rate
quaternions = list()
rates = list()

# set simulation timestep
time_step = 0.1

# create timespan
timespan = np.linspace(0, 1000, 1000) * time_step

# simulate S/C dynamics
for i in range(1000):
    
    # get AOC control inputs
    control_torque = aoc.get_control_torque(quaternion, rate)
    
    
    sc_body.propagate_states(
        external_torque=np.array([0, 0, 0]),
        cmga_torque=np.array(control_torque),
        cmga_angular_momentum=np.array([0, 0, 0]),
        time_step=time_step,
    )
    quaternion, rate = sc_body.get_states()
    quaternions.append(quaternion)
    rates.append(rate)

# plot results
fig = go.Figure()

fig.add_trace(go.Scatter(x=timespan, y=[row[0] for row in quaternions]))
fig.add_trace(go.Scatter(x=timespan, y=[row[1] for row in quaternions]))
fig.add_trace(go.Scatter(x=timespan, y=[row[2] for row in quaternions]))
fig.add_trace(go.Scatter(x=timespan, y=[row[3] for row in quaternions]))

fig.show()

fig = go.Figure()

fig.add_trace(go.Scatter(x=timespan, y=[row[0] for row in rates]))
fig.add_trace(go.Scatter(x=timespan, y=[row[1] for row in rates]))
fig.add_trace(go.Scatter(x=timespan, y=[row[2] for row in rates]))

fig.show()
