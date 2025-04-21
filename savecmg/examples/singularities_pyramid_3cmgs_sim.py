import os
import sys

# add parent directory to "sys.path" to import modules from that path
sys.path.append(os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir))

from modules.cmga import PyramidCMGA
import plotly.graph_objects as go
import numpy as np


# simulation settings
beta = [0, 90, 30, 60]
availability = [True, True, True, False]
cmgs_momenta = [1, 1, 1, 1]

# initialize CMGA object
cmga = PyramidCMGA(cmgs_beta=np.deg2rad(beta), cmgs_availability=availability, cmgs_momenta=cmgs_momenta)

# define simulation parameters
n_points = 50
det_limit = 1e-1
fig_path = "savecmg/examples/figures/" 
fig_name = "pyramid_" + "".join([str(int(x)) for x in availability]) + "_" + "_".join([str(int(x)) for x in beta]) + "_" + str(n_points) + "points.html"

singular_theta = {
    "theta_1": [],
    "theta_2": [],
    "theta_3": [],
}
singular_momentum = {
    "momentum_x": [],
    "momentum_y": [],
    "momentum_z": [],
}

# find singular values of CMGs theta and CMGA angular momentum for an array of 3 CMGs
for theta_1 in np.linspace(-np.pi, np.pi, n_points):
    print(np.rad2deg(theta_1))
    for theta_2 in np.linspace(-np.pi, np.pi, n_points):
        for theta_3 in np.linspace(-np.pi, np.pi, n_points):
            jacobian = cmga.get_jacobian(cmgs_theta=np.array([theta_1, theta_2, theta_3, 0.]))
            det = np.linalg.det(jacobian)
            if np.abs(det) < det_limit:
                momentum = cmga.get_angular_momentum(np.array([theta_1, theta_2, theta_3, 0.]))
                singular_momentum["momentum_x"].append(momentum[0])
                singular_momentum["momentum_y"].append(momentum[1])
                singular_momentum["momentum_z"].append(momentum[2])

                singular_theta["theta_1"].append(theta_1)
                singular_theta["theta_2"].append(theta_2)
                singular_theta["theta_3"].append(theta_3)

# plot results
fig1 = go.Figure(
    data=[
        go.Scatter3d(
            x=np.rad2deg(singular_theta["theta_1"]),
            y=np.rad2deg(singular_theta["theta_2"]),
            z=np.rad2deg(singular_theta["theta_3"]),
            mode="markers",
            marker=dict(size=2),
        )
    ]
)
# fig1.show()
fig1.write_html(fig_path + "angles_" + fig_name)

fig2 = go.Figure(
    data=[
        go.Scatter3d(
            x=singular_momentum["momentum_x"],
            y=singular_momentum["momentum_y"],
            z=singular_momentum["momentum_z"],
            mode="markers",
            marker=dict(size=2),
        )
    ]
)
# fig2.show()
fig2.write_html(fig_path + "momentum_" + fig_name)