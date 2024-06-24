import numpy as np
from scipy.integrate import solve_ivp


class ControlMomentGyro:
    def __init__(
        self,
        theta=0.0,
        theta_dot=0.0,
        theta_dot_max=1.5,
        angular_momentum=50.0,
        model=None,
    ):

        self.theta = theta
        self.theta_dot = theta_dot
        self.theta_dot_max = theta_dot_max
        self.angular_momentum = angular_momentum
        self.model = model

    def get_states(self):

        return self.theta, self.theta_dot, self.angular_momentum

    def propagate_states(self, theta_dot_ref, time_step):

        theta_dot_ref = np.clip(theta_dot_ref, -self.theta_dot_max, self.theta_dot_max)

        ode_solution = solve_ivp(
            fun=self._ode,
            t_span=(0.0, time_step),
            y0=[self.theta, self.theta_dot],
            method="RK45",
            dense_output=False,
            args=(theta_dot_ref,),
        )

        if (
            ode_solution.y[0][-1] <= 5 / 4 * np.pi
            and ode_solution.y[0][-1] >= -5 / 4 * np.pi
        ):

            self.theta = ode_solution.y[0][-1]
            self.theta_dot = ode_solution.y[1][-1]

        else:

            self.theta = np.sign(ode_solution.y[0][-1]) * 5 / 4 * np.pi
            self.theta_dot = 0.0

    def _ode(self, t, x, reference):

        u = reference - x[1]

        x_dot_1 = x[1]
        x_dot_2 = 10 * u
        x_dot = [x_dot_1, x_dot_2]

        return x_dot

