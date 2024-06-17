import numpy as np


class ControlMomentGyroAssembly:
    def __init__(self):

        self.beta = np.array([0.0, 0.0, 0.0, 0.0])
        self.cmg_array = [True, True, True, True]

    def get_jacobian(self, theta):

        jacobian_elements = []

        if self.cmg_array[0]:
            jacobian_elements.append(
                np.array(
                    [
                        -np.cos(self.beta[0]) * np.cos(theta[0]),
                        -np.sin(theta[0]),
                        np.sin(self.beta[0]) * np.cos(theta[0]),
                    ]
                )
            )
        if self.cmg_array[1]:
            jacobian_elements.append(
                np.array(
                    [
                        np.sin(theta[1]),
                        -np.cos(self.beta[1]) * np.cos(theta[1]),
                        np.sin(self.beta[1]) * np.cos(theta[1]),
                    ]
                )
            )
        if self.cmg_array[2]:
            jacobian_elements.append(
                np.array(
                    [
                        np.cos(theta[2]) * np.cos(theta[2]),
                        np.sin(theta[2]),
                        np.sin(self.beta[2]) * np.cos(theta[2]),
                    ]
                )
            )
        if self.cmg_array[3]:
            jacobian_elements.append(
                np.array(
                    [
                        -np.sin(theta[3]),
                        np.cos(self.beta[3]) * np.cos(theta[3]),
                        np.sin(self.beta[3]) * np.cos(theta[3]),
                    ]
                )
            )

        jacobian = np.transpose(jacobian_elements)

        return jacobian

    def get_angular_momentum(self, theta, cmgs_momenta):

        cmgs_momenta = np.delete(
            cmgs_momenta, np.where(np.array(self.cmg_array) == False)[0]
        )

        angular_momentum_elements = []

        if self.cmg_array[0]:
            angular_momentum_elements.append(
                np.array(
                    [
                        -np.cos(self.beta[0]) * np.sin(theta[0]),
                        np.cos(theta[0]),
                        np.sin(self.beta[0]) * np.sin(theta[0]),
                    ]
                )
            )
        if self.cmg_array[1]:
            angular_momentum_elements.append(
                np.array(
                    [
                        -np.cos(theta[1]),
                        -np.cos(self.beta[1]) * np.sin(theta[1]),
                        np.sin(self.beta[1]) * np.sin(theta[1]),
                    ]
                )
            )
        if self.cmg_array[2]:
            angular_momentum_elements.append(
                np.array(
                    [
                        np.cos(self.beta[2]) * np.sin(theta[2]),
                        -np.cos(theta[2]),
                        np.sin(self.beta[2]) * np.sin(theta[2]),
                    ]
                )
            )
        if self.cmg_array[3]:
            angular_momentum_elements.append(
                np.array(
                    [
                        np.cos(theta[3]),
                        np.cos(self.beta[3]) * np.sin(theta[3]),
                        np.sin(self.beta[3]) * np.sin(theta[3]),
                    ]
                )
            )

        angular_momentum = np.dot(np.transpose(angular_momentum_elements), cmgs_momenta)

        return angular_momentum

    def get_torque(self):
        pass


if __name__ == "__main__":

    import plotly.graph_objects as go

    cmga = ControlMomentGyroAssembly()
    
    # print(cmga.get_jacobian(np.array([0, 0, 0, 0])))
    # print(cmga.get_angular_momentum(np.array([0, 0, 0, 0]), [10000, 1, 1, 1]))

    beta = [0, 90, 30, 60]
    cmga.beta = np.deg2rad(beta)
    cmga.cmg_array = [False, True, True, True]
    cmgs_momenta = [1, 1, 1, 1]

    n_points = 100
    det_limit = 1e-3
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
                x=singular_theta["theta_1"],
                y=singular_theta["theta_2"],
                z=singular_theta["theta_3"],
                mode="markers",
                marker=dict(size=3),
            )
        ]
    )
    # fig1.show()
    fig1.write_html("angles_" + fig_name)

    fig2 = go.Figure(
        data=[
            go.Scatter3d(
                x=singular_momentum["momentum_x"],
                y=singular_momentum["momentum_y"],
                z=singular_momentum["momentum_z"],
                mode="markers",
                marker=dict(size=3),
            )
        ]
    )
    # fig2.show()
    fig2.write_html("momentum_" + fig_name)