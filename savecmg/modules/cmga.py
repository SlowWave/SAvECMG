import numpy as np
from .cmg import ControlMomentGyro


class ControlMomentGyroAssembly:
    def __init__(self, cmgs_beta, cmgs_availability):

        self.cmgs_beta = cmgs_beta
        self.cmgs_availability = cmgs_availability
        self.cmgs_array = None
        self.cmgs_theta = None
        self.cmgs_theta_dot = None
        self.cmgs_momenta = None
        self.jacobian = None
        self.angular_momentum = None
        self.torque = None

    def initialize_cmgs_array(
        self,
        cmgs_theta=[0.0, 0.0, 0.0, 0.0],
        cmgs_theta_dot=[0.0, 0.0, 0.0, 0.0],
        cmgs_theta_dot_max=[1.5, 1.5, 1.5, 1.5],
        cmgs_momenta=[50.0, 50.0, 50.0, 50.0],
        cmgs_model=None,
    ):

        self.cmgs_array = list()
        self.cmgs_theta = list()
        self.cmgs_theta_dot = list()
        self.cmgs_momenta = list()

        for idx, availability in enumerate(self.cmgs_availability):
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
            else:
                self.cmgs_array.append(None)
                self.cmgs_theta.append(None)
                self.cmgs_theta_dot.append(None)
                self.cmgs_momenta.append(None)

    def propagate_states(self, cmgs_theta_dot_ref, time_step):

        for idx, cmg in enumerate(self.cmgs_array):
            if cmg is not None:
                cmg.propagate_states(
                    theta_dot_ref=cmgs_theta_dot_ref[idx], time_step=time_step
                )
                cmg_states = cmg.get_states()
                self.cmgs_theta[idx] = cmg_states[0]
                self.cmgs_theta_dot[idx] = cmg_states[1]
                self.cmgs_momenta[idx] = cmg_states[2]

        self.jacobian = self.get_jacobian(self.cmgs_theta)
        self.angular_momentum = self.get_angular_momentum(
            self.cmgs_theta, self.cmgs_momenta
        )
        self.torque = self.get_torque(
            self.jacobian, self.cmgs_momenta, self.cmgs_theta_dot
        )

    def get_states(self):

        return (
            self.jacobian,
            self.angular_momentum,
            self.torque,
            self.cmgs_theta,
            self.cmgs_theta_dot,
        )

    def get_jacobian(self, cmgs_theta):

        jacobian_elements = []

        if self.cmgs_availability[0]:
            jacobian_elements.append(
                np.array(
                    [
                        -np.cos(self.cmgs_beta[0]) * np.cos(cmgs_theta[0]),
                        -np.sin(cmgs_theta[0]),
                        np.sin(self.cmgs_beta[0]) * np.cos(cmgs_theta[0]),
                    ]
                )
            )
        if self.cmgs_availability[1]:
            jacobian_elements.append(
                np.array(
                    [
                        np.sin(cmgs_theta[1]),
                        -np.cos(self.cmgs_beta[1]) * np.cos(cmgs_theta[1]),
                        np.sin(self.cmgs_beta[1]) * np.cos(cmgs_theta[1]),
                    ]
                )
            )
        if self.cmgs_availability[2]:
            jacobian_elements.append(
                np.array(
                    [
                        np.cos(cmgs_theta[2]) * np.cos(cmgs_theta[2]),
                        np.sin(cmgs_theta[2]),
                        np.sin(self.cmgs_beta[2]) * np.cos(cmgs_theta[2]),
                    ]
                )
            )
        if self.cmgs_availability[3]:
            jacobian_elements.append(
                np.array(
                    [
                        -np.sin(cmgs_theta[3]),
                        np.cos(self.cmgs_beta[3]) * np.cos(cmgs_theta[3]),
                        np.sin(self.cmgs_beta[3]) * np.cos(cmgs_theta[3]),
                    ]
                )
            )

        jacobian = np.transpose(jacobian_elements)

        return jacobian

    def get_angular_momentum(self, cmgs_theta, cmgs_momenta):

        cmgs_momenta = np.delete(
            cmgs_momenta,
            np.where(np.array(self.cmgs_availability) == False)[0],  # noqa: E712
        )

        rotation_matrix = []

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

        angular_momentum = np.dot(np.transpose(rotation_matrix), cmgs_momenta)

        return angular_momentum

    def get_torque(self, jacobian, cmgs_momenta, cmgs_theta_dot):

        cmgs_momenta = np.delete(
            cmgs_momenta,
            np.where(np.array(self.cmgs_availability) == False)[0],  # noqa: E712
        )

        cmgs_theta_dot = np.delete(
            cmgs_theta_dot,
            np.where(np.array(self.cmgs_availability) == False)[0],  # noqa: E712
        )

        torque = np.dot(np.dot(jacobian, np.diag(cmgs_momenta)), cmgs_theta_dot)

        return torque
