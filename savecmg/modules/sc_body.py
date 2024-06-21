import numpy as np
from scipy.integrate import solve_ivp


class SpacecraftBody:
    def __init__(self):

        self.quaternion = None
        self.rate = None
        self.inertia = None

    def propagate_states(
        self, external_torque, cmga_torque, cmga_angular_momentum, time_step
    ):

        ode_solution = solve_ivp(
            fun=self._ode,
            t_span=(0.0, time_step),
            y0=np.concatenate((self.quaternion, self.rate), axis=None),
            method="RK45",
            dense_output=False,
            args=(
                self.inertia,
                external_torque,
                cmga_torque,
                cmga_angular_momentum,
            ),
        )

        quaternion = [ode_solution.y[i][-1] for i in range(4)]
        self.quaternion = quaternion / np.linalg.norm(quaternion)

        self.rate = [ode_solution.y[i+4][-1] for i in range(3)]

    def get_states(self):

        return self.quaternion, self.rate

    def _ode(
        self, t, x, sc_inertia, external_torque, cmga_torque, cmga_angular_momentum
    ):

        sc_rate = np.array([x[4], x[5], x[6]])

        sc_quat_matrix = np.array(
            [
                [-x[1], -x[2], -x[3]],
                [x[0], -x[3], x[2]],
                [x[3], x[0], -x[1]],
                [-x[2], x[1], x[0]],
            ]
        )

        sc_quat_dot = np.dot(sc_quat_matrix, sc_rate) / 2

        sc_rate_dot = np.dot(
            np.linalg.inv(sc_inertia),
            (
                external_torque
                - cmga_torque
                - np.cross(
                    sc_rate, np.dot(sc_inertia, sc_rate) + cmga_angular_momentum
                )
            ),
        )

        x_dot = np.concatenate((sc_quat_dot, sc_rate_dot), axis=None)

        return x_dot


if __name__ == "__main__":

    import plotly.graph_objects as go

    sc_body = SpacecraftBody()
    sc_body.quaternion = [1, 0, 0, 0]
    sc_body.rate = [0, 0, 0]
    sc_body.inertia = np.eye(3)

    time_step = 0.1

    timespan = np.linspace(0, 1000, 1000) * time_step

    quaternion = list()
    rate = list()

    for i in range(1000):
        sc_body.propagate_states(
            external_torque=np.array([0, 0, 0]),
            cmga_torque=np.array([0, 0, 0]),
            cmga_angular_momentum=np.array([0, 0, 0]),
            time_step=time_step,
        )
        states = sc_body.get_states()
        quaternion.append(states[0])
        rate.append(states[1])

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
