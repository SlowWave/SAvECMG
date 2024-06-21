import numpy as np


class ControlMomentGyroAssembly:
    def __init__(self):

        self.cmgs_beta = np.array([0.0, 0.0, 0.0, 0.0])
        self.cmgs_array = [True, True, True, True]
        self.jacobian = None
        self.angular_momentum = None
        self.torque = None

    def propagate_states(self, cmgs_velocities_reference):
        
        # TODO: set cmgs_velocities_reference for each cmg object
        
        # TODO: propagate cmgs object states -> cmgs_velocities, cmgs_positions
        
        # TODO: compute jacobian, angular_momentum and torque in S/C reference frame
        
        # TODO: update states
        
        pass
    
    def get_states(self):
        
        return self.jacobian, self.angular_momentum, self.torque
    
    def get_jacobian(self, cmgs_theta):

        jacobian_elements = []

        if self.cmgs_array[0]:
            jacobian_elements.append(
                np.array(
                    [
                        -np.cos(self.cmgs_beta[0]) * np.cos(cmgs_theta[0]),
                        -np.sin(cmgs_theta[0]),
                        np.sin(self.cmgs_beta[0]) * np.cos(cmgs_theta[0]),
                    ]
                )
            )
        if self.cmgs_array[1]:
            jacobian_elements.append(
                np.array(
                    [
                        np.sin(cmgs_theta[1]),
                        -np.cos(self.cmgs_beta[1]) * np.cos(cmgs_theta[1]),
                        np.sin(self.cmgs_beta[1]) * np.cos(cmgs_theta[1]),
                    ]
                )
            )
        if self.cmgs_array[2]:
            jacobian_elements.append(
                np.array(
                    [
                        np.cos(cmgs_theta[2]) * np.cos(cmgs_theta[2]),
                        np.sin(cmgs_theta[2]),
                        np.sin(self.cmgs_beta[2]) * np.cos(cmgs_theta[2]),
                    ]
                )
            )
        if self.cmgs_array[3]:
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
            cmgs_momenta, np.where(np.array(self.cmgs_array) == False)[0]  # noqa: E712
        )

        rotation_matrix = []

        if self.cmgs_array[0]:
            rotation_matrix.append(
                np.array(
                    [
                        -np.cos(self.cmgs_beta[0]) * np.sin(cmgs_theta[0]),
                        np.cos(cmgs_theta[0]),
                        np.sin(self.cmgs_beta[0]) * np.sin(cmgs_theta[0]),
                    ]
                )
            )
        if self.cmgs_array[1]:
            rotation_matrix.append(
                np.array(
                    [
                        -np.cos(cmgs_theta[1]),
                        -np.cos(self.cmgs_beta[1]) * np.sin(cmgs_theta[1]),
                        np.sin(self.cmgs_beta[1]) * np.sin(cmgs_theta[1]),
                    ]
                )
            )
        if self.cmgs_array[2]:
            rotation_matrix.append(
                np.array(
                    [
                        np.cos(self.cmgs_beta[2]) * np.sin(cmgs_theta[2]),
                        -np.cos(cmgs_theta[2]),
                        np.sin(self.cmgs_beta[2]) * np.sin(cmgs_theta[2]),
                    ]
                )
            )
        if self.cmgs_array[3]:
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

    def get_torque(self, jacobian, cmgs_momenta, cmgs_velocities):
        
        cmgs_momenta = np.delete(
            cmgs_momenta, np.where(np.array(self.cmgs_array) == False)[0]  # noqa: E712
        )
        
        cmgs_velocities = np.delete(
            cmgs_velocities, np.where(np.array(self.cmgs_array) == False)[0]  # noqa: E712
        )
        
        torque = np.dot(np.dot(jacobian, np.diag(cmgs_momenta)), cmgs_velocities)
        
        return torque

