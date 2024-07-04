import os
import sys

# add parent directory to "sys.path" to import modules from that path
sys.path.append(os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir))

import numpy as np
import plotly.graph_objects as go
from modules.cmga import ControlMomentGyroAssembly
from modules.sc_body import SpacecraftBody
from modules.aoc import AttitudeController
from tqdm import tqdm

# TODO
# Update environment class
# Add script with all steering algorithms
# Set the reference quaternion in the SC step
# Class to generate quaternions based on angles input with respective plots
# Update technical notes

class Environment:
    def __init__(self):

        self.sc_body = None
        self.cmga = None
        self.aoc = None

        self.cmgs_availability = None
        self.time_step = None

        self.sim_data = {
            "time": list(),
            "sc_quat": list(),
            "sc_rate": list(),
            "sc_quat_ref": list(),
            "sc_rate_ref": None,
            "control_torque": list(),
            "cmgs_theta_dot_ref": list(),
            "cmga_jacobian": list(),
            "cmga_angular_momentum": list(),
            "cmga_torque": list(),
            "cmgs_theta": list(),
            "cmgs_theta_dot": list(),
        }

    def reset(
        self,
        cmgs_availability=[False, True, True, True],
        cmgs_beta=[0, 0, 45, 90],
        cmgs_momenta=[10, 10, 10, 10],
        sc_init_quat=[1, 0, 0, 0],
        sc_init_rate=[0, 0, 0],
        sc_quat_ref=[0, 0, 0, 1],
        sc_rate_ref=[0, 0, 0],
        sc_moi=50,
        time_step=0.1,
        cmgs_theta_dot_max=[1.5,1.5,1.5,1.5],
        aoc_k = [6,50],
        
    ):

        self.cmgs_availability = cmgs_availability
        self.time_step = time_step

        # initialize S/C
        sc_inertia = np.eye(3) * sc_moi
        self.sc_body = SpacecraftBody(sc_init_quat, sc_init_rate, sc_inertia)

        # initialize CMGA
        self.cmga = ControlMomentGyroAssembly(np.deg2rad(cmgs_beta), cmgs_availability)
        self.cmga.initialize_cmgs_array(cmgs_momenta=cmgs_momenta,cmgs_theta_dot_max=cmgs_theta_dot_max)

        # initialize AOC
        self.aoc = AttitudeController(aoc_k[0], aoc_k[1])
        self.aoc.set_reference(sc_quat_ref, sc_rate_ref)

        # get simulation data
        (
            cmga_jacobian,
            cmga_angular_momentum,
            cmga_torque,
            cmgs_theta,
            cmgs_theta_dot,
        ) = self.cmga.get_states()
        sc_quat, sc_rate = self.sc_body.get_states()
        control_torque = self.aoc.get_control_torque(sc_quat, sc_rate)

        # update simulation data
        self.sim_data["time"].append(0)
        self.sim_data["sc_quat_ref"].append(sc_quat_ref)
        self.sim_data["sc_quat"].append(sc_quat)
        self.sim_data["sc_rate"].append(sc_rate)
        self.sim_data["control_torque"].append(control_torque)
        self.sim_data["cmgs_theta_dot_ref"].append([0, 0, 0, 0])
        self.sim_data["cmga_jacobian"].append(cmga_jacobian)
        self.sim_data["cmga_angular_momentum"].append(cmga_angular_momentum)
        self.sim_data["cmga_torque"].append(cmga_torque)
        self.sim_data["cmgs_theta"].append(cmgs_theta)
        self.sim_data["cmgs_theta_dot"].append(cmgs_theta_dot)

        observation = control_torque, cmga_jacobian,                     

        return observation

    def step(self, cmgs_theta_dot_ref):

        # propagate states
        self.cmga.propagate_states(cmgs_theta_dot_ref, self.time_step)
        (
            cmga_jacobian,
            cmga_angular_momentum,
            cmga_torque,
            cmgs_theta,
            cmgs_theta_dot,
        ) = self.cmga.get_states()
        self.sc_body.propagate_states(
            np.array([0, 0, 0]), cmga_torque, cmga_angular_momentum, self.time_step
        )
        sc_quat, sc_rate = self.sc_body.get_states()
        control_torque = self.aoc.get_control_torque(sc_quat, sc_rate)

        # update simulation data
        self.sim_data["time"].append(self.sim_data["time"][-1] + self.time_step)
        self.sim_data["sc_quat_ref"].append(self.aoc.sc_quat_ref)        
        self.sim_data["sc_quat"].append(sc_quat)
        self.sim_data["sc_rate"].append(sc_rate)
        self.sim_data["control_torque"].append(control_torque)
        self.sim_data["cmgs_theta_dot_ref"].append(cmgs_theta_dot_ref)
        self.sim_data["cmga_jacobian"].append(cmga_jacobian)
        self.sim_data["cmga_angular_momentum"].append(cmga_angular_momentum)
        self.sim_data["cmga_torque"].append(cmga_torque)
        self.sim_data["cmgs_theta"].append(cmgs_theta)
        self.sim_data["cmgs_theta_dot"].append(cmgs_theta_dot)

        observation = control_torque, cmga_jacobian

        return observation

    def plot_sim_data(self):
        colors =['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # S/C quaternions
        fig = go.Figure()
        fig.update_layout(title="S/C quaternions", xaxis_title="time", yaxis_title="quaternion")
        for idx in range(4):
            fig.add_trace(
                go.Scatter(
                    x=self.sim_data["time"],
                    y=[row[idx] for row in self.sim_data["sc_quat"]], name="q_{}".format(idx),
                    line=dict(color=colors[idx % len(colors)])

                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.sim_data["time"],
                    y=[row[idx] for row in self.sim_data["sc_quat_ref"]], name="q_ref_{}".format(idx),
                    line=dict(dash='dot',color=colors[idx % len(colors)])

                )
            )
        fig.show()

        # S/C rates
        fig = go.Figure()
        fig.update_layout(title="S/C rates", xaxis_title="time", yaxis_title="rate [rad/s]")
        for idx in range(3):
            fig.add_trace(
                go.Scatter(
                    x=self.sim_data["time"],
                    y=[row[idx] for row in self.sim_data["sc_rate"]], name="w_{}".format(idx),
                    line=dict(color=colors[idx % len(colors)])

                )
            )
        fig.show()

        # CMGs theta_dot
        fig = go.Figure()
        fig.update_layout(title="CMGs theta_dot", xaxis_title="time", yaxis_title="theta_dot [rad/s]")
        
        for idx, available in enumerate(self.cmgs_availability):
            if available:
                fig.add_trace(
                    go.Scatter(
                        x=self.sim_data["time"],
                        y=[row[idx] for row in self.sim_data["cmgs_theta_dot_ref"]], name="theta_dot_ref_{}".format(idx),
                        mode='lines', line=dict(dash='dot',color=colors[idx % len(colors)]),
                    )
                )                
                fig.add_trace(
                    go.Scatter(
                        x=self.sim_data["time"],
                        y=[row[idx] for row in self.sim_data["cmgs_theta_dot"]], name="theta_dot_{}".format(idx),
                        line=dict(color=colors[idx % len(colors)])
                    )
                )
        
        fig.show()

        # CMGs theta
        fig = go.Figure()
        fig.update_layout(title="CMGs theta", xaxis_title="time", yaxis_title="theta [deg]")
        for idx, available in enumerate(self.cmgs_availability):
            if available:          
                fig.add_trace(
                    go.Scatter(
                        x=self.sim_data["time"],
                        y=[np.rad2deg(row[idx]) for row in self.sim_data["cmgs_theta"]],
                        name="theta_{}".format(idx),
                        line=dict(color=colors[idx % len(colors)])
                        
                    )
                )
        fig.show()

        # CMGA torque
        fig = go.Figure()
        fig.update_layout(title="CMGA torque", xaxis_title="time", yaxis_title="torque [Nm]")
        for idx in range(3):
            fig.add_trace(
                go.Scatter(
                    x=self.sim_data["time"],
                    y=[row[idx] for row in self.sim_data["control_torque"]], name="control_torque_{}".format(idx),
                    mode='lines', line=dict(dash='dot',color=colors[idx % len(colors)])
                )
            )                
            fig.add_trace(
                go.Scatter(
                    x=self.sim_data["time"],
                    y=[row[idx] for row in self.sim_data["cmga_torque"]], name="cmga_torque_{}".format(idx),
                    line=dict(color=colors[idx % len(colors)])

                )
            )
        fig.show()


if __name__ == "__main__":

    def simple_control(cmgs_availability, manip_idx, cmga_jacobian, control_torque, cmgs_theta_dot_prev):

        try:
            manip_idx.append(np.sqrt(np.abs(np.linalg.det(np.dot(cmga_jacobian, cmga_jacobian.T)))))

            cmgs_theta_dot_ref = np.dot(
                np.linalg.pinv(cmga_jacobian),
                control_torque,
            )
        except Exception:
            print("singular matrix")
            return cmgs_theta_dot_prev

        cmgs_theta_dot_ref_iter = iter(cmgs_theta_dot_ref)
        full_cmgs_theta_dot_ref = list()

        for cmg in cmgs_availability:
            if cmg:
                full_cmgs_theta_dot_ref.append(next(cmgs_theta_dot_ref_iter))
            else:
                full_cmgs_theta_dot_ref.append(0)
            
        return full_cmgs_theta_dot_ref

    def robust_control(cmgs_availability, manip_idx_list,alpha_list, cmga_jacobian, control_torque, cmgs_theta_dot_prev):

        alpha0 = 1
        ni = 0.002
        manip_idx = np.sqrt(np.abs(np.linalg.det(np.dot(cmga_jacobian, cmga_jacobian.T))))
        alpha = alpha0 * np.exp(-ni * manip_idx)
        # print(alpha)
        
        manip_idx_list.append(manip_idx)
        alpha_list.append(alpha)

        jacob_quad = np.dot(cmga_jacobian, cmga_jacobian.T)
        inv_term = np.linalg.inv(jacob_quad + alpha * np.eye(3))
        pseudo_jacob_inv = np.dot(cmga_jacobian.T, inv_term)
        
        cmgs_theta_dot_ref = np.dot(pseudo_jacob_inv, control_torque)

        cmgs_theta_dot_ref_iter = iter(cmgs_theta_dot_ref)
        full_cmgs_theta_dot_ref = list()

        for cmg in cmgs_availability:
            if cmg:
                full_cmgs_theta_dot_ref.append(next(cmgs_theta_dot_ref_iter))
            else:
                full_cmgs_theta_dot_ref.append(0)
            
        return full_cmgs_theta_dot_ref

    def null_torque_control(cmgs_availability, manip_idx_list, cmga_jacobian, control_torque, derivative_jacobian,null_torque_theta_dot_list,null_torque,alpha_list):

        alpha0 = 0.000003
        ni = 0.002
        manip_idx = np.sqrt(np.abs(np.linalg.det(np.dot(cmga_jacobian, cmga_jacobian.T))))
        alpha = alpha0 * np.exp(-ni * manip_idx)
        # alpha = 0.000003
        # print(alpha)
        alpha_list.append(alpha)
        manip_idx_list.append(manip_idx)

        jacob_quad = np.dot(cmga_jacobian, cmga_jacobian.T)
        # inv_term = np.linalg.inv(jacob_quad + alpha * np.eye(3))
        inv_term = np.linalg.inv(jacob_quad)
        pseudo_jacob_inv = np.dot(cmga_jacobian.T, inv_term)
        B = (np.eye(4) - np.dot(pseudo_jacob_inv, cmga_jacobian))
        
        
        null_torque_theta_dot = np.array(alpha * np.dot(B, derivative_jacobian),dtype=np.float32)
        null_torque_theta_dot_list.append(null_torque_theta_dot)
        null_torque.append(np.dot(cmga_jacobian,null_torque_theta_dot))
        # print(null_torque_theta_dot)
        
        cmgs_theta_dot_ref = np.dot(pseudo_jacob_inv, control_torque) + null_torque_theta_dot

        cmgs_theta_dot_ref_iter = iter(cmgs_theta_dot_ref)
        full_cmgs_theta_dot_ref = list()

        for cmg in cmgs_availability:
            if cmg:
                full_cmgs_theta_dot_ref.append(next(cmgs_theta_dot_ref_iter))
            else:
                full_cmgs_theta_dot_ref.append(0)
            
        return full_cmgs_theta_dot_ref
    def null_robust_torque_control(cmgs_availability, manip_idx_list, cmga_jacobian, control_torque, derivative_jacobian,null_torque_theta_dot_list,null_torque,alpha_list):

        alpha0 = 1
        ni = 0.002
        manip_idx = np.sqrt(np.abs(np.linalg.det(np.dot(cmga_jacobian, cmga_jacobian.T))))
        alpha = alpha0 * np.exp(-ni * manip_idx)
        beta0 = 0.000003
        ni = 0.002
        manip_idx = np.sqrt(np.abs(np.linalg.det(np.dot(cmga_jacobian, cmga_jacobian.T))))
        beta = beta0 * np.exp(-ni * manip_idx)
        # print(alpha)
        alpha_list.append(alpha)
        manip_idx_list.append(manip_idx)

        jacob_quad = np.dot(cmga_jacobian, cmga_jacobian.T)
        # inv_term = np.linalg.inv(jacob_quad + alpha * np.eye(3))
        inv_term = np.linalg.inv(jacob_quad+ alpha * np.eye(3))
        pseudo_jacob_inv = np.dot(cmga_jacobian.T, inv_term)
        
        
        null_torque_theta_dot = np.array(beta * np.dot((np.eye(4) - np.dot(np.linalg.pinv(cmga_jacobian), cmga_jacobian)), derivative_jacobian),dtype=np.float32)
        null_torque_theta_dot_list.append(null_torque_theta_dot)
        null_torque.append(np.dot(cmga_jacobian,null_torque_theta_dot))
        # print(null_torque_theta_dot)
        
        cmgs_theta_dot_ref = np.dot(pseudo_jacob_inv, control_torque) + null_torque_theta_dot

        cmgs_theta_dot_ref_iter = iter(cmgs_theta_dot_ref)
        full_cmgs_theta_dot_ref = list()

        for cmg in cmgs_availability:
            if cmg:
                full_cmgs_theta_dot_ref.append(next(cmgs_theta_dot_ref_iter))
            else:
                full_cmgs_theta_dot_ref.append(0)
            
        return full_cmgs_theta_dot_ref


    time_step = 0.1

    env = Environment()
    cmgs_availability = [True, True, False, True]
    control_torque, cmga_jacobian = env.reset(
        cmgs_availability=cmgs_availability,
        cmgs_beta=[60, 60, 60, 60],
        sc_quat_ref=[-1, -1, -1, -1],
        sc_moi=400,
        time_step=time_step,
        cmgs_momenta=[10,10,10,10],
        cmgs_theta_dot_max=[1.5,1.5,1.5,1.5],
        
    )
    manip_idx = list()
    alpha_list = list()
    null_torque_theta_dot = list()
    null_torque = list()
    derivative_jacobian_list = list()
    cmgs_theta_dot_ref = [0, 0, 0, 0]

    for _ in tqdm(range(3000)):
        
        # cmgs_theta_dot_ref = simple_control(cmgs_availability, manip_idx, cmga_jacobian, control_torque, cmgs_theta_dot_ref)
        cmgs_theta = env.sim_data["cmgs_theta"][-1]
        derivative_jacobian = env.cmga.substitute_to_symbolic_jacobian(cmgs_theta)
        derivative_jacobian_list.append(derivative_jacobian)
        cmgs_theta_dot_ref = robust_control(cmgs_availability, manip_idx, alpha_list, cmga_jacobian, control_torque, cmgs_theta_dot_ref)
        # cmgs_theta_dot_ref = null_torque_control(cmgs_availability, manip_idx, cmga_jacobian, control_torque, derivative_jacobian,null_torque_theta_dot,null_torque,alpha_list)
        # cmgs_theta_dot_ref = null_robust_torque_control(cmgs_availability, manip_idx, cmga_jacobian, control_torque, derivative_jacobian,null_torque_theta_dot,null_torque,alpha_list)


        control_torque, cmga_jacobian = env.step(cmgs_theta_dot_ref)

    env.plot_sim_data()

    fig = go.Figure()
    fig.update_layout(title="manipulability index")
    fig.add_trace(go.Scatter(x=env.sim_data["time"],y=manip_idx))
    fig.show()
    fig = go.Figure()
    fig.update_layout(title="alpha index")
    fig.add_trace(go.Scatter(x=env.sim_data["time"],y=alpha_list))
    fig.show()
    fig = go.Figure()
    fig.update_layout(title="null torque speed reference")
    for idx in range(4):
        fig.add_trace(go.Scatter(x=env.sim_data["time"],y=[row[idx] for row in null_torque_theta_dot], name="w_ref_null_{}".format(idx)))
    fig.show()
    fig = go.Figure()
    fig.update_layout(title="jacobian  derivative")
    for idx in range(4):
        fig.add_trace(go.Scatter(x=env.sim_data["time"],y=[row[idx] for row in derivative_jacobian_list], name="dJ_dtheta{}".format(idx)))
    fig.show()
    fig = go.Figure()
    fig.update_layout(title="null torque ")
    for idx in range(3):
        fig.add_trace(go.Scatter(x=env.sim_data["time"],y=[row[idx] for row in null_torque], name="T_ref_null_{}".format(idx)))
    fig.show()