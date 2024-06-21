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


if __name__ == "__main__":

    import plotly.graph_objects as go

    cmg = ControlMomentGyro(
        theta=0, theta_dot=0, theta_dot_max=0.1, angular_momentum=0, model=0
    )

    theta = list()
    theta_dot = list()
    time_step = 0.1

    timespan = np.linspace(0, 1000, 1000) * time_step

    for i in range(1000):
        cmg.propagate_states(theta_dot_ref=-0.3, time_step=time_step)
        states = cmg.get_states()
        theta.append(states[0])
        theta_dot.append(states[1])

    fig_1 = go.Figure()
    fig_1.add_trace(go.Scatter(x=timespan, y=theta))
    fig_1.show()

    fig_2 = go.Figure()
    fig_2.add_trace(go.Scatter(x=timespan, y=theta_dot))
    fig_2.show()
