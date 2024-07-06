import numpy as np
import plotly.graph_objects as go

class NullTorqueSteering():
    def __init__(self, cmgs_availability, beta0, ni):
        
        self.cmgs_availability = cmgs_availability
        self.beta0 = beta0
        self.ni = ni
        
        self.null_torque_list = list()
        self.null_torque_theta_dot_list = list()

    def get_cmgs_theta_dot_ref(self, aoc_control_torque, cmga_jacobian, cmga_manip_idx, cmga_manip_idx_grad):
        
        beta = self.beta0 * np.exp(-self.ni * cmga_manip_idx)
        
        jacob_quad = np.dot(cmga_jacobian, cmga_jacobian.T)
        inv_term = np.linalg.inv(jacob_quad)
        pseudo_jacob_inv = np.dot(cmga_jacobian.T, inv_term)
        
        null_space_map = (np.eye(4) - np.dot(pseudo_jacob_inv, cmga_jacobian))
        
        null_torque_theta_dot = np.array(beta * np.dot(null_space_map, cmga_manip_idx_grad), dtype=np.float32)
        self.null_torque_theta_dot_list.append(null_torque_theta_dot)
        self.null_torque_list.append(np.dot(cmga_jacobian, null_torque_theta_dot))
        
        cmgs_theta_dot_ref = np.dot(pseudo_jacob_inv, aoc_control_torque) + null_torque_theta_dot
        
        cmgs_theta_dot_ref_iter = iter(cmgs_theta_dot_ref)
        full_cmgs_theta_dot_ref = list()

        for cmg in self.cmgs_availability:
            if cmg:
                full_cmgs_theta_dot_ref.append(next(cmgs_theta_dot_ref_iter))
            else:
                full_cmgs_theta_dot_ref.append(0)
            
        return full_cmgs_theta_dot_ref
    
    
    def plot_null_torque(self):
        
        fig = go.Figure()
        fig.update_layout(
            title="CMGA Null Torque", yaxis_title="torque [Nm]"
        )
        for idx in range(3):
            fig.add_trace(
                go.Scatter(
                    y=[row[idx] for row in self.null_torque_list],
                    name="T_{}".format(idx),
                )
            )
        fig.show()
        
    def plot_null_torque_theta_dot(self):
        
        fig = go.Figure()
        fig.update_layout(
            title="CMGA Null Torque theta_dot", yaxis_title="theta_dot [rad/s]"
        )
        for idx in range(4):
            fig.add_trace(
                go.Scatter(
                    y=[row[idx] for row in self.null_torque_theta_dot_list],
                    name="w_{}".format(idx),
                )
            )
        fig.show()