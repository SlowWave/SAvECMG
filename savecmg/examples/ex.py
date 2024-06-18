import os
import sys

# add parent directory to "sys.path" to import modules from that path
sys.path.append(os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir))

from modules.cmga import ControlMomentGyroAssembly
import plotly.graph_objects as go
import numpy as np

cmga = ControlMomentGyroAssembly()

beta = [0, 90, 30, 60]
cmga.beta = np.deg2rad(beta)
cmga.cmg_array = [False, True, True, True]
cmgs_momenta = [1, 1, 1, 1]

n_points = 361
det_limit = 1e-3
fig_path = "savecmg/examples/figures/" 
fig_name = str(beta[1]) + "_" + str(beta[2]) + "_" + str(beta[3]) + "_" + str(n_points) + "pnts.html"

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

for theta_1 in np.linspace(-np.pi, np.pi, n_points):
    print(np.rad2deg(theta_1))
    for theta_2 in np.linspace(-np.pi, np.pi, n_points):
        for theta_3 in np.linspace(-np.pi, np.pi, n_points):
            jacobian = cmga.get_jacobian(np.array([0., theta_1, theta_2, theta_3]))
            det = np.linalg.det(jacobian)
            if np.abs(det) < det_limit:
                # print(f"singular: {np.rad2deg(theta_1)}, {np.rad2deg(theta_2)}, {np.rad2deg(theta_3)}")
                momentum = cmga.get_angular_momentum(np.array([0., theta_1, theta_2, theta_3]), cmgs_momenta)
                singular_momentum["momentum_x"].append(momentum[0])
                singular_momentum["momentum_y"].append(momentum[1])
                singular_momentum["momentum_z"].append(momentum[2])

                singular_theta["theta_1"].append(theta_1)
                singular_theta["theta_2"].append(theta_2)
                singular_theta["theta_3"].append(theta_3)

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