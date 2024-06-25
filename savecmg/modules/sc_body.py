import numpy as np
from scipy.integrate import solve_ivp


class SpacecraftBody:
    def __init__(self, quaternion, rate, inertia):
        """
        Initializes a new instance of the SpacecraftBody class.

        Args:
            quaternion (list, array): A list or numpy array of length 4 representing the quaternion of the spacecraft body.
            rate (list, array): A list or numpy array of length 3 representing the rate of the spacecraft body.
            inertia (list, array): A list of lists ([[row1], [row2], [row3]]) or numpy array of shape (3, 3) representing the inertia matrix of the spacecraft body.

        Returns:
            None
        """

        self.quaternion = quaternion / np.linalg.norm(quaternion)
        self.rate = np.array(rate)
        self.inertia = np.array(inertia)

    def propagate_states(
        self, external_torque, cmga_torque, cmga_angular_momentum, time_step
    ):
        """
        Propagates the states of the spacecraft body over a given time step.

        Args:
            external_torque (list, array): A list or numpy array of length 3 representing the external disturbance torque acting on the spacecraft body.
            cmga_torque (list, array): A list or numpy array of length 3 representing the torque acting on the spacecraft body due to the Control Moment Gyro Assembly.
            cmga_angular_momentum (list, array): A list or numpy array of length 3 representing the angular momentum of the spacecraft body due to the Control Moment Gyro Assembly.
            time_step (float): The time step over which to propagate the states.

        Returns:
            None
        """

        # propagate spacecraft dynamics
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

        # update spacecraft states
        quaternion = [ode_solution.y[i][-1] for i in range(4)]
        self.quaternion = quaternion / np.linalg.norm(quaternion)
        self.rate = [ode_solution.y[i + 4][-1] for i in range(3)]

    def get_states(self):
        """
        Returns the current states of the spacecraft body.

        Returns:
            tuple: A tuple containing the quaternion and rate of the spacecraft body.
        """

        return self.quaternion, self.rate

    def _ode(
        self, t, x, sc_inertia, external_torque, cmga_torque, cmga_angular_momentum
    ):
        """
        Computes the derivative of the spacecraft body state variables with respect to time.

        Args:
            t (float): The current time.
            x (numpy.ndarray): The current state variables of the spacecraft body.
            sc_inertia (numpy.ndarray): The inertia matrix of the spacecraft body.
            external_torque (numpy.ndarray): The external torque acting on the spacecraft body.
            cmga_torque (numpy.ndarray): The torque from the control moment gyro assemblies (CMGAs).
            cmga_angular_momentum (numpy.ndarray): The angular momentum of the CMGAs.

        Returns:
            numpy.ndarray: The derivative of the state variables with respect to time.
        """

        # unpack state variables
        sc_rate = np.array([x[4], x[5], x[6]])

        sc_quat_matrix = np.array(
            [
                [-x[1], -x[2], -x[3]],
                [x[0], -x[3], x[2]],
                [x[3], x[0], -x[1]],
                [-x[2], x[1], x[0]],
            ]
        )

        # compute state derivatives
        sc_quat_dot = np.dot(sc_quat_matrix, sc_rate) / 2

        sc_rate_dot = np.dot(
            np.linalg.inv(sc_inertia),
            (
                external_torque
                - cmga_torque
                - np.cross(sc_rate, np.dot(sc_inertia, sc_rate) + cmga_angular_momentum)
            ),
        )

        # concatenate state derivatives
        x_dot = np.concatenate((sc_quat_dot, sc_rate_dot), axis=None)

        return x_dot
