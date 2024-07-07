import numpy as np

class SimpleSteering():
    def __init__(self, cmgs_availability):

        self.cmgs_availability = cmgs_availability

    def get_cmgs_theta_dot_ref(self, aoc_control_torque, cmga_jacobian):
        
        jacob_quad = np.dot(cmga_jacobian, cmga_jacobian.T)
        inv_term = np.linalg.inv(jacob_quad)
        pseudo_jacob_inv = np.dot(cmga_jacobian.T, inv_term)

        cmgs_theta_dot_ref = np.dot(pseudo_jacob_inv, aoc_control_torque)

        cmgs_theta_dot_ref_iter = iter(cmgs_theta_dot_ref)
        full_cmgs_theta_dot_ref = list()

        for cmg in self.cmgs_availability:
            if cmg:
                full_cmgs_theta_dot_ref.append(next(cmgs_theta_dot_ref_iter))
            else:
                full_cmgs_theta_dot_ref.append(0)

        return full_cmgs_theta_dot_ref

    