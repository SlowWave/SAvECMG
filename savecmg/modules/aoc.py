import numpy as np


class AttitudeController:
    def __init__(self, k_quat=1, k_rate=1):
        """
        Initializes the AttitudeController with the specified quaternion and rate gains.
        
        Args:
            k_quat (float): The quaternion gain value. Defaults to 1.
            k_rate (float): The rate gain value. Defaults to 1.
            
        Returns:
            None
        """

        self.k_quat = k_quat
        self.k_rate = k_rate
        self.sc_quat_ref = None
        self.sc_rate_ref = None

    def set_reference(self, sc_quat_ref, sc_rate_ref):
        """
        Set the reference quaternion and rate for the spacecraft.

        Args:
            sc_quat_ref (array): The reference quaternion.
            sc_rate_ref (array): The reference rate.

        Returns:
            None
        """

        # normalize the reference quaternion
        self.sc_quat_ref = sc_quat_ref / np.linalg.norm(sc_quat_ref)
        self.sc_rate_ref = np.array(sc_rate_ref)

    def get_control_torque(self, sc_quat, sc_rate):
        """
        Computes the control torque required to achieve the desired attitude.

        Args:
            sc_quat (array): The current quaternion representation of the spacecraft's attitude.
            sc_rate (array): The current spacecraft rate.

        Returns:
            array: The control torque required to achieve the desired attitude.
        """

        # normalize the current quaternion
        sc_quat = sc_quat / np.linalg.norm(sc_quat)
        sc_rate = np.array(sc_rate)

        # compute conjugate of the reference quaternion
        sc_quat_ref_conj = np.array(
            [
                self.sc_quat_ref[0],
                -self.sc_quat_ref[1],
                -self.sc_quat_ref[2],
                -self.sc_quat_ref[3],
            ]
        )

        # compute error between current and reference quaternion
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

        # compute error between current and reference rate
        sc_rate_err = self.sc_rate_ref - sc_rate

        # compute control torque
        control_torque = -self.k_quat * sc_quat_err[1:4] - self.k_rate * sc_rate_err

        return control_torque
