import numpy as np


class AttitudeController:
    def __init__(self, k_quat, k_rate):

        self.k_quat = k_quat
        self.k_rate = k_rate
        self.sc_quat_ref = None
        self.sc_rate_ref = None

    def set_reference(self, sc_quat_ref, sc_rate_ref):

        self.sc_quat_ref = sc_quat_ref / np.linalg.norm(sc_quat_ref)
        self.sc_rate_ref = np.array(sc_rate_ref)

    def get_control_torque(self, sc_quat, sc_rate):

        sc_quat = sc_quat / np.linalg.norm(sc_quat)
        sc_rate = np.array(sc_rate)

        sc_quat_ref_conj = np.array(
            [
                self.sc_quat_ref[0],
                -self.sc_quat_ref[1],
                -self.sc_quat_ref[2],
                -self.sc_quat_ref[3],
            ]
        )

        sc_quat_err = np.array(
            [
                +sc_quat[0] * sc_quat_ref_conj[0]
                - sc_quat[1] * sc_quat_ref_conj[1]
                - sc_quat[2] * sc_quat_ref_conj[2]
                - sc_quat[3] * sc_quat_ref_conj[3],
                sc_quat[0] * sc_quat_ref_conj[1]
                + sc_quat[1] * sc_quat_ref_conj[0]
                - sc_quat[2] * sc_quat_ref_conj[3]
                + sc_quat[3] * sc_quat_ref_conj[2],
                sc_quat[0] * sc_quat_ref_conj[2]
                + sc_quat[1] * sc_quat_ref_conj[3]
                + sc_quat[2] * sc_quat_ref_conj[0]
                - sc_quat[3] * sc_quat_ref_conj[1],
                sc_quat[0] * sc_quat_ref_conj[3]
                - sc_quat[1] * sc_quat_ref_conj[2]
                + sc_quat[2] * sc_quat_ref_conj[1]
                + sc_quat[3] * sc_quat_ref_conj[0],
            ]
        )

        sc_rate_err = self.sc_rate_ref - sc_rate

        control_torque = -self.k_quat * sc_quat_err[1:4] - self.k_rate * sc_rate_err

        return control_torque
