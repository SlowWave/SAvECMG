import numpy as np
from scipy.integrate import solve_ivp


class SpacecraftBody:
    def __init__(self, quaternion, rate, inertia):

        self.quaternion = quaternion
        self.rate = rate
        self.inertia = inertia

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

