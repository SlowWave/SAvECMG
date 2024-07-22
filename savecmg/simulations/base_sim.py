import os
import sys

# add parent directory to "sys.path" to import modules from that path
sys.path.append(os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir))

from envs.base_env import Environment
from steering_algos.singularity_escape import SingularityRobustInverse
from steering_algos.simple_steering import SimpleSteering

# define simulation parameters
cmgs_availability = [False, True, True, True]
cmgs_beta = [0, 30, 60, 90]
cmgs_momenta = [10, 10, 10, 10]
sc_quat_init = [-1, -1, -1, 1]
sc_rate_init = [0, 0, 0]
sc_quat_ref = [0, 0, 0, 1]
sc_rate_ref = [0, 0, 0]
sc_moi = 500
time_step = 0.5
cmgs_theta_dot_max = [1.5, 1.5, 1.5, 1.5]
aoc_gains = [1, 10]

# initialize simulation environment
sim_env = Environment()
aoc_control_torque, cmga_jacobian, cmga_manip_idx, cmga_manip_idx_grad = sim_env.reset(
    cmgs_availability=cmgs_availability,
    cmgs_beta=cmgs_beta,
    cmgs_momenta=cmgs_momenta,
    cmgs_theta_dot_max=cmgs_theta_dot_max,
    sc_quat_init=sc_quat_init,
    sc_rate_init=sc_rate_init,
    sc_moi=sc_moi,
    sc_quat_ref=sc_quat_ref,
    sc_rate_ref=sc_rate_ref,
    aoc_gains=aoc_gains,
    time_step=time_step,
)

# define steering algorithm parameters
alpha0 = 1
ni = 0.002

# initialize steering algorithm
sri_steering = SingularityRobustInverse(
    cmgs_availability=cmgs_availability,
    alpha0=alpha0,
    ni=ni
)

for _ in range(1000):
    
    cmgs_theta_dot_ref = sri_steering.get_cmgs_theta_dot_ref(
        aoc_control_torque=aoc_control_torque,
        cmga_jacobian=cmga_jacobian,
        cmga_manip_idx=cmga_manip_idx
    )
    
    aoc_control_torque, cmga_jacobian, cmga_manip_idx, cmga_manip_idx_grad = sim_env.step(cmgs_theta_dot_ref)


sim_env.plot_sim_data()