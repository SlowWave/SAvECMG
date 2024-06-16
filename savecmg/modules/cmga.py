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

        jacobian = np.array(jacobian_elements)

        return jacobian

    def get_angular_momentum(self, theta, cmgs_momenta):

        
        del(cmgs_momenta[np.where(np.array(np.array(self.cmg_array)) == False)[0]])
        cmgs_momenta = np.array(cmgs_momenta)

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

        angular_momentum = np.dot(np.array(angular_momentum_elements), cmgs_momenta)

        return angular_momentum

    def get_torque(self):
        pass


if __name__ == "__main__":
    
    cmga = ControlMomentGyroAssembly()
    
    print(cmga.get_jacobian(np.array([0, 0, 0, 0])))
    
    print(cmga.get_angular_momentum(np.array([0, 0, 0, 0]), [10000, 1, 1, 1]))