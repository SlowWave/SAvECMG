import os
import sys

# add parent directory to "sys.path" to import modules from that path
sys.path.append(os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir))

import numpy as np
import plotly.graph_objects as go
from modules.cmga import ControlMomentGyroAssembly
from modules.sc_body import SpacecraftBody
from modules.aoc import AttitudeController


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
            "sc_rate_ref": list(),
            "control_torque": list(),
            "cmgs_theta_dot_ref": list(),
            "cmga_angular_momentum": list(),
            "cmga_torque": list(),
            "cmga_jacobian": list(),
            "cmga_manip_idx": list(),
            "cmga_manip_idx_grad": list(),
            "cmgs_theta": list(),
            "cmgs_theta_dot": list(),
        }

    def reset(
        self,
        cmgs_availability=[False, True, True, True],
        cmgs_beta=[0, 0, 45, 90],
        cmgs_momenta=[10, 10, 10, 10],
        cmgs_theta_dot_max=[1.5, 1.5, 1.5, 1.5],
        sc_quat_init=[1, 0, 0, 0],
        sc_rate_init=[0, 0, 0],
        sc_moi=500,
        sc_quat_ref=[0, 0, 0, 1],
        sc_rate_ref=[0, 0, 0],
        aoc_gains=[1, 10],
        time_step=0.5,
    ):

        self.cmgs_availability = cmgs_availability
        self.time_step = time_step

        # initialize S/C
        sc_inertia = np.eye(3) * sc_moi
        self.sc_body = SpacecraftBody(sc_quat_init, sc_rate_init, sc_inertia)

        # initialize CMGA
        self.cmga = ControlMomentGyroAssembly(np.deg2rad(cmgs_beta), cmgs_availability)
        self.cmga.initialize_cmgs_array(
            cmgs_momenta=cmgs_momenta, cmgs_theta_dot_max=cmgs_theta_dot_max
        )

        # initialize AOC
        self.aoc = AttitudeController(aoc_gains[0], aoc_gains[1])
        self.aoc.set_reference(sc_quat_ref, sc_rate_ref)

        # get simulation data
        cmga_states = self.cmga.get_states()
        sc_quat, sc_rate = self.sc_body.get_states()
        control_torque = self.aoc.get_control_torque(sc_quat, sc_rate)

        # update simulation data
        self.sim_data["time"].append(0)
        self.sim_data["sc_quat"].append(sc_quat)
        self.sim_data["sc_rate"].append(sc_rate)
        self.sim_data["sc_quat_ref"].append(sc_quat_ref)
        self.sim_data["sc_rate_ref"].append(sc_rate_ref)
        self.sim_data["control_torque"].append(control_torque)
        self.sim_data["cmgs_theta_dot_ref"].append([0, 0, 0, 0])
        self.sim_data["cmga_angular_momentum"].append(cmga_states["angular_momentum"])
        self.sim_data["cmga_torque"].append(cmga_states["torque"])
        self.sim_data["cmga_jacobian"].append(cmga_states["jacobian"])
        self.sim_data["cmga_manip_idx"].append(cmga_states["manip_idx"])
        self.sim_data["cmga_manip_idx_grad"].append(cmga_states["manip_idx_gradient"])
        self.sim_data["cmgs_theta"].append(cmga_states["cmgs_theta"])
        self.sim_data["cmgs_theta_dot"].append(cmga_states["cmgs_theta_dot"])

        observation = control_torque, cmga_states["jacobian"], cmga_states["manip_idx"], cmga_states["manip_idx_gradient"]

        return observation

    def step(self, cmgs_theta_dot_ref):

        # propagate states
        self.cmga.propagate_states(cmgs_theta_dot_ref, self.time_step)
        cmga_states = self.cmga.get_states()

        self.sc_body.propagate_states(
            np.array([0, 0, 0]),
            cmga_states["torque"],
            cmga_states["angular_momentum"],
            self.time_step,
        )
        sc_quat, sc_rate = self.sc_body.get_states()
        control_torque = self.aoc.get_control_torque(sc_quat, sc_rate)

        # update simulation data
        self.sim_data["time"].append(self.sim_data["time"][-1] + self.time_step)
        self.sim_data["sc_quat"].append(sc_quat)
        self.sim_data["sc_rate"].append(sc_rate)
        self.sim_data["sc_quat_ref"].append(self.aoc.sc_quat_ref)
        self.sim_data["sc_rate_ref"].append(self.aoc.sc_rate_ref)
        self.sim_data["control_torque"].append(control_torque)
        self.sim_data["cmgs_theta_dot_ref"].append(cmgs_theta_dot_ref)
        self.sim_data["cmga_angular_momentum"].append(cmga_states["angular_momentum"])
        self.sim_data["cmga_torque"].append(cmga_states["torque"])
        self.sim_data["cmga_jacobian"].append(cmga_states["jacobian"])
        self.sim_data["cmga_manip_idx"].append(cmga_states["manip_idx"])
        self.sim_data["cmga_manip_idx_grad"].append(cmga_states["manip_idx_gradient"])
        self.sim_data["cmgs_theta"].append(cmga_states["cmgs_theta"])
        self.sim_data["cmgs_theta_dot"].append(cmga_states["cmgs_theta_dot"])

        observation = control_torque, cmga_states["jacobian"], cmga_states["manip_idx"], cmga_states["manip_idx_gradient"]

        return observation

    def plot_sim_data(
        self,
        plot_sc_quat=True,
        plot_sc_rate=True,
        plot_cmga_torque=True,
        plot_cmga_angular_momentum=True,
        plot_cmga_manip_idx=True,
        plot_cmgs_theta=True,
        plot_cmgs_theta_dot=True,
    ):

        color_palette = [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
        ]


        # S/C quaternions
        if plot_sc_quat:
            fig = go.Figure()
            fig.update_layout(
                title="S/C quaternions", xaxis_title="time [s]", yaxis_title="quaternion"
            )
            for idx in range(4):
                fig.add_trace(
                    go.Scatter(
                        x=self.sim_data["time"],
                        y=[row[idx] for row in self.sim_data["sc_quat"]],
                        name="q_{}".format(idx),
                        line=dict(color=color_palette[idx % len(color_palette)]),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=self.sim_data["time"],
                        y=[row[idx] for row in self.sim_data["sc_quat_ref"]],
                        name="q_ref_{}".format(idx),
                        line=dict(
                            dash="dot", color=color_palette[idx % len(color_palette)]
                        ),
                    )
                )
            fig.show()

        # S/C rates
        if plot_sc_rate:
            fig = go.Figure()
            fig.update_layout(
                title="S/C rates", xaxis_title="time [s]", yaxis_title="rate [rad/s]"
            )
            for idx in range(3):
                fig.add_trace(
                    go.Scatter(
                        x=self.sim_data["time"],
                        y=[row[idx] for row in self.sim_data["sc_rate"]],
                        name="w_{}".format(idx),
                        line=dict(color=color_palette[idx % len(color_palette)]),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=self.sim_data["time"],
                        y=[row[idx] for row in self.sim_data["sc_rate_ref"]],
                        name="w_ref_{}".format(idx),
                        line=dict(
                            dash="dot", color=color_palette[idx % len(color_palette)]
                        ),
                    )
                )
            fig.show()

        # CMGA torque
        if plot_cmga_torque:
            fig = go.Figure()
            fig.update_layout(
                title="CMGA torque", xaxis_title="time [s]", yaxis_title="torque [Nm]"
            )
            for idx in range(3):
                fig.add_trace(
                    go.Scatter(
                        x=self.sim_data["time"],
                        y=[row[idx] for row in self.sim_data["control_torque"]],
                        name="control_torque_{}".format(idx),
                        mode="lines",
                        line=dict(
                            dash="dot", color=color_palette[idx % len(color_palette)]
                        ),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=self.sim_data["time"],
                        y=[row[idx] for row in self.sim_data["cmga_torque"]],
                        name="cmga_torque_{}".format(idx),
                        line=dict(color=color_palette[idx % len(color_palette)]),
                    )
                )
            fig.show()

        # CMGA manip_idx
        if plot_cmga_manip_idx:
            fig = go.Figure()
            fig.update_layout(
                title="CMGA manip_idx", xaxis_title="time [s]", yaxis_title="manip_idx"
            )
            fig.add_trace(
                go.Scatter(
                    x=self.sim_data["time"],
                    y=self.sim_data["cmga_manip_idx"],
                    name="manip_idx",
                    line=dict(color=color_palette[0]),
                )
            )
            fig.show()

        # CMGs theta
        if plot_cmgs_theta:
            fig = go.Figure()
            fig.update_layout(
                title="CMGs theta", xaxis_title="time [s]", yaxis_title="theta [deg]"
            )
            for idx, available in enumerate(self.cmgs_availability):
                if available:
                    fig.add_trace(
                        go.Scatter(
                            x=self.sim_data["time"],
                            y=[np.rad2deg(row[idx]) for row in self.sim_data["cmgs_theta"]],
                            name="theta_{}".format(idx),
                            line=dict(color=color_palette[idx % len(color_palette)]),
                        )
                    )
            fig.show()

        # CMGs theta_dot
        if plot_cmgs_theta_dot:
            fig = go.Figure()
            fig.update_layout(
                title="CMGs theta_dot", xaxis_title="time [s]", yaxis_title="theta_dot [rad/s]"
            )

            for idx, available in enumerate(self.cmgs_availability):
                if available:
                    fig.add_trace(
                        go.Scatter(
                            x=self.sim_data["time"],
                            y=[row[idx] for row in self.sim_data["cmgs_theta_dot_ref"]],
                            name="theta_dot_ref_{}".format(idx),
                            mode="lines",
                            line=dict(
                                dash="dot", color=color_palette[idx % len(color_palette)]
                            ),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=self.sim_data["time"],
                            y=[row[idx] for row in self.sim_data["cmgs_theta_dot"]],
                            name="theta_dot_{}".format(idx),
                            line=dict(color=color_palette[idx % len(color_palette)]),
                        )
                    )

            fig.show()

