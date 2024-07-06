import numpy as np

class SingularityRobustInverse():
    def __init__(self, cmgs_availability, alpha0, ni):
        
        self.cmgs_availability = cmgs_availability
        self.alpha0 = alpha0
        self.ni = ni

    def get_cmgs_theta_dot_ref(self, aoc_control_torque, cmga_jacobian, cmga_manip_idx):
        
        alpha = self.alpha0 * np.exp(-self.ni * cmga_manip_idx)
        
        jacob_quad = np.dot(cmga_jacobian, cmga_jacobian.T)
        inv_term = np.linalg.inv(jacob_quad + alpha * np.eye(3))
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
    
    