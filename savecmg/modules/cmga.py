import numpy as np
from scipy.optimize import fsolve
from .cmg import ControlMomentGyro
import sympy as sym

# TODO: add another CMG configuration (object?)


class ControlMomentGyroAssembly:
    def __init__(
        self, cmgs_beta, cmgs_availability, cmgs_momenta=[10.0, 10.0, 10.0, 10.0]
    ):
        """
        Initializes a ControlMomentGyroAssembly object.

        Args:
            cmgs_beta (List[float]): A list of beta angles for the control moment gyroscopes [rad].
            cmgs_availability (List[bool]): A list of booleans indicating the availability of each control moment gyroscope.
            cmgs_momenta (List[float], optional): A list of momenta for the control moment gyros [N]. Defaults to [10.0, 10.0, 10.0, 10.0].

        Returns:
            None
        """

        # initialize parameters
        self.cmgs_beta = cmgs_beta
        self.cmgs_availability = cmgs_availability
        self.cmgs_momenta = cmgs_momenta
        self.cmgs_array = None
        self.cmgs_theta = None
        self.cmgs_theta_dot = None
        self.angular_momentum = None
        self.torque = None
        self.jacobian = None
        self.manip_idx = None
        self.manip_idx_gradient = np.array([0, 0, 0, 0])

        # initialize symbolic functions
        self._symbolic_jacobian = self._compute_symbolic_jacobian()
        self._symbolic_manip_idx = self._compute_symbolic_manip_idx()
        self._symbolic_manip_idx_gradient = self._compute_symbolic_manip_idx_gradient()

    def initialize_cmgs_array(
        self,
        set_zero_momentum=True,
        cmgs_theta=[0.0, 0.0, 0.0, 0.0],
        cmgs_theta_dot=[0.0, 0.0, 0.0, 0.0],
        cmgs_theta_dot_max=[1.5, 1.5, 1.5, 1.5],
        cmgs_momenta=[50.0, 50.0, 50.0, 50.0],
        cmgs_model=None,
    ):
        """
        Initializes the control moment gyroscopes array with the given parameters.

        Args:
            set_zero_momentum (bool, optional): Whether to set zero momentum configuration for the control moment gyroscopes. If True, the initial values of CMGs theta are chosen to have zero angular momentum. Defaults to True.
            cmgs_theta (List[float], optional): Initial values of theta angles for the control moment gyroscopes [rad]. Defaults to [0.0, 0.0, 0.0, 0.0].
            cmgs_theta_dot (List[float], optional): Initial values of theta_dot angles for the control moment gyroscopes [rads]. Defaults to [0.0, 0.0, 0.0, 0.0].
            cmgs_theta_dot_max (List[float], optional): Maximum values of theta_dot velocities for the control moment gyroscopes [rads]. Defaults to [1.5, 1.5, 1.5, 1.5].
            cmgs_momenta (List[float], optional): Angular momenta values for the control moment gyroscopes [Nms]. Defaults to [50.0, 50.0, 50.0, 50.0].
            cmgs_model (str, optional): The model of the control moment gyroscope. Defaults to None.

        Returns:
            None
        """

        self.cmgs_array = list()
        self.cmgs_theta = list()
        self.cmgs_theta_dot = list()
        self.cmgs_momenta = list()

        if set_zero_momentum:
            cmgs_theta = self._compute_cmgs_theta_zero_momentum()
            cmgs_theta_dot = [0.0, 0.0, 0.0, 0.0]

        # populate cmgs_array
        for idx, availability in enumerate(self.cmgs_availability):
            # if cmgs is available initialize a CMG object and append it to the cmgs_array
            if availability:
                self.cmgs_array.append(
                    ControlMomentGyro(
                        theta=cmgs_theta[idx],
                        theta_dot=cmgs_theta_dot[idx],
                        theta_dot_max=cmgs_theta_dot_max[idx],
                        angular_momentum=cmgs_momenta[idx],
                        model=cmgs_model,
                    )
                )
                self.cmgs_theta.append(cmgs_theta[idx])
                self.cmgs_theta_dot.append(cmgs_theta_dot[idx])
                self.cmgs_momenta.append(cmgs_momenta[idx])
            # if cmgs is not available append None to the cmgs_array
            else:
                self.cmgs_array.append(None)
                self.cmgs_theta.append(None)
                self.cmgs_theta_dot.append(None)
                self.cmgs_momenta.append(None)

    def propagate_states(self, cmgs_theta_dot_ref, time_step):
        """
        Propagates the states CMGs array based on the given reference theta_dot and time step.

        Args:
            cmgs_theta_dot_ref (List[float]): A list of reference theta_dot values for the CMGs.
            time_step (float): The time step for the propagation.

        Returns:
            None
        """

        cmgs_theta = [None, None, None, None]
        cmgs_theta_dot = [None, None, None, None]
        cmgs_momenta = [None, None, None, None]

        # propagate CMGs dynamics for each available CMG in the cmgs_array
        for idx, cmg in enumerate(self.cmgs_array):
            if cmg is not None:
                cmg.propagate_states(
                    theta_dot_ref=cmgs_theta_dot_ref[idx], time_step=time_step
                )

                # get CMGs states
                cmg_states = cmg.get_states()
                cmgs_theta[idx] = cmg_states[0]
                cmgs_theta_dot[idx] = cmg_states[1]
                cmgs_momenta[idx] = cmg_states[2]

        #
        cmgs_theta_0 = np.copy(self.cmgs_theta)
        cmgs_theta_1 = np.copy(cmgs_theta)
        manip_idx_0 = np.copy(self.manip_idx)

        # update CMGs states
        self.cmgs_theta = cmgs_theta
        self.cmgs_theta_dot = cmgs_theta_dot
        self.cmgs_momenta = cmgs_momenta

        # update CMGA states
        self.angular_momentum = self.get_angular_momentum()
        self.torque = self.get_torque()
        self.jacobian = self.get_jacobian()
        self.manip_idx = self.get_manip_idx()
        
        manip_idx_1 = np.copy(self.manip_idx)
        self.manip_idx_gradient = self.get_manip_idx_gradient(manip_idx_0, manip_idx_1, cmgs_theta_0, cmgs_theta_1)

    def get_states(self):
        """
        Returns the current CMGA states.

        Returns:
            dict: A dictionary containing the current CMGA states.
        """

        # update CMGA states
        self.angular_momentum = self.get_angular_momentum()
        self.torque = self.get_torque()
        self.jacobian = self.get_jacobian()
        self.manip_idx = self.get_manip_idx()

        states = dict(
            cmgs_theta=self.cmgs_theta,
            cmgs_theta_dot=self.cmgs_theta_dot,
            cmgs_momenta=self.cmgs_momenta,
            angular_momentum=self.angular_momentum,
            torque=self.torque,
            jacobian=self.jacobian,
            manip_idx=self.manip_idx,
        )

        return states

    def get_angular_momentum(self, cmgs_theta=None):
        """
        Computes the CMGA angular momentum in S/C body frame based on the given CMGs theta angles.

        Args:
            cmgs_theta (List[float]): A list of CMGs theta angles [rad].

        Returns:
            ndarray: CMGA angular momentum in S/C body frame [Nms].
        """

        if not cmgs_theta:
            cmgs_theta = self.cmgs_theta

        # remove useless CMGs momenta based on CMGs availability
        cmgs_momenta = np.delete(
            self.cmgs_momenta,
            np.where(np.array(self.cmgs_availability) == False)[0],  # noqa: E712
        ).tolist()

        rotation_matrix = []

        # compute rotation matrix elements
        if self.cmgs_availability[0]:
            rotation_matrix.append(
                np.array(
                    [
                        -np.cos(self.cmgs_beta[0]) * np.sin(cmgs_theta[0]),
                        np.cos(cmgs_theta[0]),
                        np.sin(self.cmgs_beta[0]) * np.sin(cmgs_theta[0]),
                    ]
                )
            )
        if self.cmgs_availability[1]:
            rotation_matrix.append(
                np.array(
                    [
                        -np.cos(cmgs_theta[1]),
                        -np.cos(self.cmgs_beta[1]) * np.sin(cmgs_theta[1]),
                        np.sin(self.cmgs_beta[1]) * np.sin(cmgs_theta[1]),
                    ]
                )
            )
        if self.cmgs_availability[2]:
            rotation_matrix.append(
                np.array(
                    [
                        np.cos(self.cmgs_beta[2]) * np.sin(cmgs_theta[2]),
                        -np.cos(cmgs_theta[2]),
                        np.sin(self.cmgs_beta[2]) * np.sin(cmgs_theta[2]),
                    ]
                )
            )
        if self.cmgs_availability[3]:
            rotation_matrix.append(
                np.array(
                    [
                        np.cos(cmgs_theta[3]),
                        np.cos(self.cmgs_beta[3]) * np.sin(cmgs_theta[3]),
                        np.sin(self.cmgs_beta[3]) * np.sin(cmgs_theta[3]),
                    ]
                )
            )

        # compute CMGA angular momentum in S/C fixed frame
        angular_momentum = np.dot(np.transpose(rotation_matrix), cmgs_momenta)

        return angular_momentum

    def get_torque(self, cmgs_theta_dot=None):
        """
        Computes the torque based on the given CMGs angular velocities.

        Args:
            cmgs_theta_dot (List[float]): CMGs angular velocities array [rads].

        Returns:
            CMGA Torque in S/C body frame [Nm].
        """

        if not cmgs_theta_dot:
            cmgs_theta_dot = self.cmgs_theta_dot

        # remove useless CMGs theta_dot based on CMGs availability
        cmgs_theta_dot = np.delete(
            cmgs_theta_dot,
            np.where(np.array(self.cmgs_availability) == False)[0],  # noqa: E712
        ).tolist()

        # compute CMGA torque in S/C fixed frame
        torque = np.dot(np.array(self.jacobian, dtype=np.float32), cmgs_theta_dot)

        return torque

    def get_jacobian(self, symbolic=False, cmgs_theta=None):
        """
        Computes the CMGA Jacobian matrix based on the given CMGs theta angles.

        Args:
            cmgs_theta (List[float]): A list of CMGs theta angles [rad].

        Returns:
            ndarray: The Jacobian matrix calculated based on the input angles.
        """

        if not symbolic:

            if not cmgs_theta:
                cmgs_theta = self.cmgs_theta

            jacobian_elements = []

            # compute jacobian matrix elements
            if self.cmgs_availability[0]:
                jacobian_elements.append(
                    self.cmgs_momenta[0]
                    * np.array(
                        [
                            -np.cos(self.cmgs_beta[0]) * np.cos(cmgs_theta[0]),
                            -np.sin(cmgs_theta[0]),
                            np.sin(self.cmgs_beta[0]) * np.cos(cmgs_theta[0]),
                        ]
                    )
                )
            if self.cmgs_availability[1]:
                jacobian_elements.append(
                    self.cmgs_momenta[1]
                    * np.array(
                        [
                            np.sin(cmgs_theta[1]),
                            -np.cos(self.cmgs_beta[1]) * np.cos(cmgs_theta[1]),
                            np.sin(self.cmgs_beta[1]) * np.cos(cmgs_theta[1]),
                        ]
                    )
                )
            if self.cmgs_availability[2]:
                jacobian_elements.append(
                    self.cmgs_momenta[2]
                    * np.array(
                        [
                            np.cos(self.cmgs_beta[2]) * np.cos(cmgs_theta[2]),
                            np.sin(cmgs_theta[2]),
                            np.sin(self.cmgs_beta[2]) * np.cos(cmgs_theta[2]),
                        ]
                    )
                )
            if self.cmgs_availability[3]:
                jacobian_elements.append(
                    self.cmgs_momenta[3]
                    * np.array(
                        [
                            -np.sin(cmgs_theta[3]),
                            np.cos(self.cmgs_beta[3]) * np.cos(cmgs_theta[3]),
                            np.sin(self.cmgs_beta[3]) * np.cos(cmgs_theta[3]),
                        ]
                    )
                )

            # jacobian matrix is a 3xn matrix where n is the number of available CMGs (1<=n<=4)
            jacobian = np.transpose(jacobian_elements)

        else:

            theta_1 = sym.Symbol("theta_1")
            theta_2 = sym.Symbol("theta_2")
            theta_3 = sym.Symbol("theta_3")
            theta_4 = sym.Symbol("theta_4")

            jacobian_function = sym.lambdify(
                (theta_1, theta_2, theta_3, theta_4), self._symbolic_jacobian, "numpy"
            )
            jacobian = jacobian_function(
                cmgs_theta[0], cmgs_theta[1], cmgs_theta[2], cmgs_theta[3]
            )

        return jacobian

    def get_manip_idx(self, symbolic=False, cmgs_theta=None):

        if not symbolic:

            manip_idx = np.sqrt(
                np.abs(np.linalg.det(np.dot(self.jacobian, self.jacobian.T)))
            )

        else:

            if not cmgs_theta:
                cmgs_theta = self.cmgs_theta

            theta_1 = sym.Symbol("theta_1")
            theta_2 = sym.Symbol("theta_2")
            theta_3 = sym.Symbol("theta_3")
            theta_4 = sym.Symbol("theta_4")

            manip_idx_function = sym.lambdify(
                (theta_1, theta_2, theta_3, theta_4), self._symbolic_manip_idx, "numpy"
            )
            manip_idx = manip_idx_function(
                cmgs_theta[0], cmgs_theta[1], cmgs_theta[2], cmgs_theta[3]
            )

        return manip_idx

    def get_manip_idx_gradient(self, manip_idx_0=None, manip_idx_1=None, cmgs_theta_0=None, cmgs_theta_1=None, symbolic=False):

        if not symbolic:

            if self.manip_idx is None or self.cmgs_theta is None:
                manip_idx_grad = np.array([0, 0, 0, 0])
            else:
                epsilon = 0.0001
                delta_manip_idx = - (manip_idx_1 ** 1/7 - manip_idx_0 ** 1/7)
                delta_theta = np.array(
                    [
                        abs(cmgs_theta_1[0] - cmgs_theta_0[0]) + epsilon,
                        abs(cmgs_theta_1[1] - cmgs_theta_0[1]) + epsilon,
                        abs(cmgs_theta_1[2] - cmgs_theta_0[2]) + epsilon,
                        abs(cmgs_theta_1[3] - cmgs_theta_0[3]) + epsilon,
                    ]
                )

                manip_idx_grad = delta_manip_idx / delta_theta
                # manip_idx_grad = np.where(delta_manip_idx < 0, 0, delta_manip_idx / delta_theta)
                # print(delta_manip_idx, delta_theta, manip_idx_grad)

        else:

            if not cmgs_theta_1:
                cmgs_theta = self.cmgs_theta
            else:
                cmgs_theta = cmgs_theta_1

            theta_1 = sym.Symbol("theta_1")
            theta_2 = sym.Symbol("theta_2")
            theta_3 = sym.Symbol("theta_3")
            theta_4 = sym.Symbol("theta_4")

            manip_idx_grad_function = sym.lambdify(
                (theta_1, theta_2, theta_3, theta_4),
                self._symbolic_manip_idx_gradient,
                "numpy",
            )
            manip_idx_grad = manip_idx_grad_function(
                cmgs_theta[0], cmgs_theta[1], cmgs_theta[2], cmgs_theta[3]
            )

        return manip_idx_grad

    def _compute_symbolic_jacobian(self):

        theta_1 = sym.Symbol("theta_1")
        theta_2 = sym.Symbol("theta_2")
        theta_3 = sym.Symbol("theta_3")
        theta_4 = sym.Symbol("theta_4")

        jacobian_elements = []

        # compute jacobian matrix elements
        if self.cmgs_availability[0]:
            jacobian_elements.append(
                self.cmgs_momenta[0]
                * np.array(
                    [
                        -np.cos(self.cmgs_beta[0]) * sym.cos(theta_1),
                        -sym.sin(theta_1),
                        np.sin(self.cmgs_beta[0]) * sym.cos(theta_1),
                    ]
                )
            )
        if self.cmgs_availability[1]:
            jacobian_elements.append(
                self.cmgs_momenta[1]
                * np.array(
                    [
                        sym.sin(theta_2),
                        -np.cos(self.cmgs_beta[1]) * sym.cos(theta_2),
                        np.sin(self.cmgs_beta[1]) * sym.cos(theta_2),
                    ]
                )
            )
        if self.cmgs_availability[2]:
            jacobian_elements.append(
                self.cmgs_momenta[2]
                * np.array(
                    [
                        np.cos(self.cmgs_beta[2]) * sym.cos(theta_3),
                        sym.sin(theta_3),
                        np.sin(self.cmgs_beta[2]) * sym.cos(theta_3),
                    ]
                )
            )
        if self.cmgs_availability[3]:
            jacobian_elements.append(
                self.cmgs_momenta[3]
                * np.array(
                    [
                        -sym.sin(theta_4),
                        np.cos(self.cmgs_beta[3]) * sym.cos(theta_4),
                        np.sin(self.cmgs_beta[3]) * sym.cos(theta_4),
                    ]
                )
            )

        # jacobian matrix is a 3xn matrix where n is the number of available CMGs (1<=n<=4)
        symbolic_jacobian = sym.Matrix(np.transpose(jacobian_elements))

        return symbolic_jacobian

    def _compute_symbolic_manip_idx(self):

        jacobian_det = sym.Matrix(
            np.dot(self._symbolic_jacobian, self._symbolic_jacobian.T)
        ).det()
        symbolic_manip_idx = sym.sqrt(sym.sqrt((jacobian_det) ** 2))

        return symbolic_manip_idx

    def _compute_symbolic_manip_idx_gradient(self):

        theta_1 = sym.Symbol("theta_1")
        theta_2 = sym.Symbol("theta_2")
        theta_3 = sym.Symbol("theta_3")
        theta_4 = sym.Symbol("theta_4")

        symbolic_theta = sym.Array([theta_1, theta_2, theta_3, theta_4])
        manip_idx_gradient = -sym.diff(self._symbolic_manip_idx**2, symbolic_theta)

        return manip_idx_gradient

    def _compute_cmgs_theta_zero_momentum(self):

        if sum(self.cmgs_availability) == 2:
            initial_guess = [0, 0]
        elif sum(self.cmgs_availability) == 3:
            initial_guess = [0, 0, 0]
        elif sum(self.cmgs_availability) == 4:
            return [0, 0, 0, 0]

        solution = iter(
            fsolve(
                self.__zero_momentum_equations,
                initial_guess,
                args=(self.cmgs_beta, self.cmgs_availability),
            )
        )

        theta = list()
        for cmg in self.cmgs_availability:
            if cmg:
                theta.append(next(solution))
            else:
                theta.append(0)

        return theta

    def __zero_momentum_equations(self, vars, beta, cmgs_availability):

        match cmgs_availability:

            # 4 CMGs
            case [True, True, True, True]:
                k = [1, 1, 1, 1]
                theta_4 = 0
                theta_1, theta_2, theta_3 = vars
            # 3 CMGs
            case [False, True, True, True]:
                k = [0, 1, 1, 1]
                theta_1 = 0
                theta_2, theta_3, theta_4 = vars
            case [True, False, True, True]:
                k = [1, 0, 1, 1]
                theta_2 = 0
                theta_1, theta_3, theta_4 = vars
            case [True, True, False, True]:
                k = [1, 1, 0, 1]
                theta_3 = 0
                theta_1, theta_2, theta_4 = vars
            case [True, True, True, False]:
                k = [1, 1, 1, 0]
                theta_4 = 0
                theta_1, theta_2, theta_3 = vars
            # 2 CMGs
            case [True, True, False, False]:
                k = [1, 1, 0, 0]
                theta_3 = 0
                theta_4 = 0
                theta_1, theta_2 = vars
            case [True, False, True, False]:
                k = [1, 0, 1, 0]
                theta_2 = 0
                theta_4 = 0
                theta_1, theta_3 = vars
            case [True, False, False, True]:
                k = [1, 0, 0, 1]
                theta_2 = 0
                theta_3 = 0
                theta_1, theta_4 = vars
            case [False, True, True, False]:
                k = [0, 1, 0, 1]
                theta_1 = 0
                theta_4 = 0
                theta_2, theta_3 = vars
            case [False, True, False, True]:
                k = [0, 1, 0, 1]
                theta_1 = 0
                theta_3 = 0
                theta_2, theta_4 = vars
            case [False, False, True, True]:
                k = [0, 0, 1, 1]
                theta_1 = 0
                theta_2 = 0
                theta_3, theta_4 = vars

        eq_1 = (
            -k[0] * np.cos(beta[0]) * np.sin(theta_1)
            - k[1] * np.cos(theta_2)
            + k[2] * np.cos(beta[2]) * np.sin(theta_3)
            + k[3] * np.cos(theta_4)
        )
        eq_2 = (
            k[0] * np.cos(theta_1)
            - k[1] * np.cos(beta[1]) * np.sin(theta_2)
            - k[2] * np.cos(theta_3)
            + k[3] * np.cos(beta[3]) * np.sin(theta_4)
        )
        eq_3 = (
            k[0] * np.sin(beta[0]) * np.sin(theta_1)
            + k[1] * np.sin(beta[1]) * np.sin(theta_2)
            + k[2] * np.sin(beta[2]) * np.sin(theta_3)
            + k[3] * np.sin(beta[3]) * np.sin(theta_4)
        )

        if sum(k) == 2:
            eqs = [eq_1, eq_2]
        else:
            eqs = [eq_1, eq_2, eq_3]

        return eqs
