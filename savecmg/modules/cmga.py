import numpy as np


class ControlMomentGyroAssembly:
    def __init__(self):

        self.beta = np.array([0.0, 0.0, 0.0, 0.0])
        self.cmg_array = [True, True, True, True]
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
        
        # TODO: return states (jacobian, angular_momentum, torque)
        
        pass
    
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

        rotation_matrix = []

        if self.cmg_array[0]:
            rotation_matrix.append(
                np.array(
                    [
                        -np.cos(self.beta[0]) * np.sin(theta[0]),
                        np.cos(theta[0]),
                        np.sin(self.beta[0]) * np.sin(theta[0]),
                    ]
                )
            )
        if self.cmg_array[1]:
            rotation_matrix.append(
                np.array(
                    [
                        -np.cos(theta[1]),
                        -np.cos(self.beta[1]) * np.sin(theta[1]),
                        np.sin(self.beta[1]) * np.sin(theta[1]),
                    ]
                )
            )
        if self.cmg_array[2]:
            rotation_matrix.append(
                np.array(
                    [
                        np.cos(self.beta[2]) * np.sin(theta[2]),
                        -np.cos(theta[2]),
                        np.sin(self.beta[2]) * np.sin(theta[2]),
                    ]
                )
            )
        if self.cmg_array[3]:
            rotation_matrix.append(
                np.array(
                    [
                        np.cos(theta[3]),
                        np.cos(self.beta[3]) * np.sin(theta[3]),
                        np.sin(self.beta[3]) * np.sin(theta[3]),
                    ]
                )
            )

        angular_momentum = np.dot(np.transpose(rotation_matrix), cmgs_momenta)

        return angular_momentum

    def get_torque(self, jacobian, cmgs_velocities):
        pass

