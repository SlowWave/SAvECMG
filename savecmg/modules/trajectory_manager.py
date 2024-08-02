import os
import sys

# add parent directory to "sys.path" to import modules from that path
sys.path.append(os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir))
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from sympy import cos, sin
from scipy.optimize import minimize
import sympy as sym
from sympy import lambdify
from modules.trajectory_generator import TrajectoryGenerator
from tqdm import tqdm
class TrajectoryManager():
    def __init__(self,sequence):
        self.sequence = sequence
        self.output_trajectory = {"traj_pos_1":list(),
                                 "traj_pos_2":list(),
                                 "traj_pos_3":list(),
                                 "traj_vel_1":list(),
                                 "traj_vel_2":list(),
                                 "traj_vel_3":list(),
                                 }
        self.output_quat = []



    def wrap_trajectories(self,traj_type=1,trajectory_1=None,trajectory_2=None,trajectory_3=None):
        """It wraps from one to three trajectories into one matrix

        Args:
            type (bool, 0 for speed e 1 for position)
            trajectory_1 (list/array [1xn], float)
            trajectory_2 (list/array [1xn], float)
            trajectory_3 (list/array [1xn], float)


        Returns:
            dict (6 List [1xn], float): traj_pos_1,traj_pos_2,traj_pos_3,traj_vel_1,traj_vel_2,traj_vel_3
        """
        if trajectory_1 == None:
            trajectory_1 = list()
        if trajectory_2 == None:
            trajectory_2 = list()
        if trajectory_3 == None:
            trajectory_3 = list()

        if type(trajectory_1) == list:
            pass
        else:
            trajectory_1 = trajectory_1.tolist()
        if type(trajectory_2) == list:
            pass
        else:
            trajectory_2 = trajectory_2.tolist()
        if type(trajectory_3) == list:
            pass
        else:
            trajectory_3 = trajectory_3.tolist()

        if traj_type:
            traj_string = 'traj_pos_'
        else: 
            traj_string = 'traj_vel_'

        keep_traj_1 = 0
        keep_traj_2 = 0
        keep_traj_3 = 0
        keep_traj_1 += trajectory_1[-1] if trajectory_1  else 0
        keep_traj_2 += trajectory_2[-1] if trajectory_2  else 0
        keep_traj_3 += trajectory_3[-1] if trajectory_3  else 0
        keep_traj_1 += self.output_trajectory[traj_string+'1'][-1] if self.output_trajectory[traj_string+'1']  else 0
        keep_traj_2 += self.output_trajectory[traj_string+'2'][-1] if self.output_trajectory[traj_string+'2']  else 0
        keep_traj_3 += self.output_trajectory[traj_string+'3'][-1] if self.output_trajectory[traj_string+'3']   else 0


        n1 = len(trajectory_1)
        n2 = len(trajectory_2)
        n3 = len(trajectory_3)
        n_max = max(n1,n2,n3)
        traj_1 = trajectory_1 + (keep_traj_1*np.ones(n_max-n1)).tolist()
        traj_2 = trajectory_2 + (keep_traj_2*np.ones(n_max-n2)).tolist()
        traj_3 = trajectory_3 + (keep_traj_3*np.ones(n_max-n3)).tolist()
        
        self.output_trajectory[traj_string+'1'].extend(traj_1)
        self.output_trajectory[traj_string+'2'].extend(traj_2)
        self.output_trajectory[traj_string+'3'].extend(traj_3)

        return self.output_trajectory



    def __rotation_matrix(self, axis, theta):
        if axis.lower() == 'x':
            return sym.Matrix([[1, 0, 0], [0, cos(theta), -sin(theta)], [0, sin(theta), cos(theta)]])
        elif axis.lower() == 'y':
            return sym.Matrix([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
        elif axis.lower() == 'z':
            return sym.Matrix([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])
        
    def __compute_symbolic_W(self):
        theta_vec = sym.symbols('theta_vec1:4')
        theta_vec_dot = sym.symbols('theta_vec_dot1:4')
        theta_1, theta_2, theta_3 = theta_vec
        theta_dot_1, theta_dot_2, theta_dot_3 = theta_vec_dot
        

        # Rotation matrices based on the sequence
        R_1 = self.__rotation_matrix(self.sequence[0], theta_1)
        R_2 = self.__rotation_matrix(self.sequence[1], theta_2)
        R_3 = self.__rotation_matrix(self.sequence[2], theta_3)

        

        R_eul = R_1 @ R_2 @ R_3
        dR_1 = sym.diff(R_1,theta_1)*theta_dot_1
        dR_2 = sym.diff(R_2,theta_2)*theta_dot_2
        dR_3 = sym.diff(R_3,theta_3)*theta_dot_3
        dR_eul = dR_1@R_2@R_3 + R_1@dR_2@R_3 + R_1@R_2@dR_3
        dR_eul = R_eul.T@dR_eul 
        omega_1 = dR_eul[2,1]
        omega_2 = dR_eul[0,2]
        omega_3 = dR_eul[1,0]
        return sym.Array([omega_1,omega_2,omega_3])
        
        
        
    def __compute_symbolic_R(self, shape):
        theta_vec = sym.symbols('theta_vec1:4')
        theta_1, theta_2, theta_3 = theta_vec

        # Rotation matrices based on the sequence
        R_1 = self.__rotation_matrix(self.sequence[0], theta_1)
        R_2 = self.__rotation_matrix(self.sequence[1], theta_2)
        R_3 = self.__rotation_matrix(self.sequence[2], theta_3)

        R_eul = R_1 @ (R_2 @ R_3)

        q = sym.symbols('q1:5')
        q_0, q_1, q_2, q_3 = q

        R_quat = sym.Matrix([
            [q_0**2 + q_1**2 - q_2**2 - q_3**2, 2*(q_1*q_2 - q_0*q_3), 2*(q_1*q_3 + q_0*q_2)],
            [2*(q_1*q_2 + q_0*q_3), q_0**2 - q_1**2 + q_2**2 - q_3**2, 2*(q_2*q_3 - q_0*q_1)],
            [2*(q_1*q_3 - q_0*q_2), 2*(q_2*q_3 + q_0*q_1), q_0**2 - q_1**2 - q_2**2 + q_3**2]
        ])
        
        constraints = [
            (R_eul[0, 0] - R_quat[0, 0])**2,
            (R_eul[0, 1] - R_quat[0, 1])**2,
            (R_eul[0, 2] - R_quat[0, 2])**2,
            (R_eul[1, 0] - R_quat[1, 0])**2,
            (R_eul[1, 1] - R_quat[1, 1])**2,
            (R_eul[1, 2] - R_quat[1, 2])**2,
            (R_eul[2, 0] - R_quat[2, 0])**2,
            (R_eul[2, 1] - R_quat[2, 1])**2,
            (R_eul[2, 2] - R_quat[2, 2])**2,
            # q_0**2 + q_1**2 + q_2**2 + q_3**2 - 1  # Normalization constraint
        ]
        

        f_scalar = np.sum(constraints)
        return f_scalar


    def _lambdify_eul_to_quat(self):
        # It converts the symbol equation into function (quaternion unkwown variable)
        theta_vec = sym.symbols('theta_vec1:4')
        q = sym.symbols('q1:5')
        eq = self.__compute_symbolic_R(shape=4)
        f_eq_function = lambdify([q,theta_vec],eq,'numpy')
        return f_eq_function

    def _lambdify_quat_to_eul(self):
        # It converts the symbol equation into function (theta unkwown variable)
        theta_vec = sym.symbols('theta_vec1:4')
        q = sym.symbols('q1:5')
        eq = self.__compute_symbolic_R(shape=3)
        f_eq_function = lambdify([theta_vec,q],eq,'numpy')
        return f_eq_function
    
    def _lambdify_eul_to_omega(self):
        theta_vec = sym.symbols('theta_vec1:4')
        theta_vec_dot = sym.symbols('theta_vec_dot1:4')        
        omega_vec = self.__compute_symbolic_W()
        omega_vec_function = lambdify([theta_vec,theta_vec_dot],omega_vec,'numpy')
        return omega_vec_function
    
    
    def eul_to_omega(self,theta,theta_dot):
        theta = np.array(theta)
        if theta.shape[0] != 3:
            theta = theta.T
        theta_dot = np.array(theta_dot)
        if theta_dot.shape[0] != 3:
            theta_dot = theta_dot.T
        omega_vec_function = self._lambdify_eul_to_omega()
        omega_vec = []
        for i in tqdm(range(theta.shape[1])):        
            omega = omega_vec_function(theta[:,i],theta_dot[:,i])
            omega_vec.append(omega)
        return np.array(omega_vec).T
             
        



    def eul_to_quat(self, theta):

        """It converts an array of euler angles into an array of quaternion

        Args:
            eul angles (array[3xn] ,float)

        Returns:
            quat (array[4xn] ,float)
        """
        theta = np.array(theta)
        if theta.shape[0] != 3:
            theta = theta.T

        eq = self._lambdify_eul_to_quat()
        q_solution = []
        q_solution_library = []
        q_0 = np.array([1, 0, 0, 0])
        options = {'maxiter': 500}

        for angles in tqdm(theta.T):
            res = minimize(eq, x0=q_0, args=(angles,), method='SLSQP', options=options,tol=1e-15)
            q_solution.append(res.x)
            q_solution_library.append(R.from_euler(self.sequence,angles.tolist(),degrees=False).as_quat())
            
            q_0 = res.x 
            q_0 /= np.linalg.norm(q_0)
        self.output_quat = np.array(q_solution).T

        return np.array(q_solution).T


    def quat_to_eul(self, quat):
        """It converts an array of quaternion into an array of euler angles

        Args:
            quat (array[4xn] ,float)

        Returns:
            eul angles (array[3xn] ,float)
        """
        quat = np.array(quat)
        if quat.shape[0] != 4:
            quat = quat.T

        eq = self._lambdify_quat_to_eul()
        theta_solution = []
        theta_solution_library = []
        options = { 'maxiter': 500}

        for i, q in enumerate(tqdm(quat.T)):
            theta_0 = [0, 0, 0] if i == 0 else theta_solution[-1] + (np.random.rand(1)-0.5)*2
            constraints = [
                {'type': 'ineq', 'fun': lambda vars: 2*np.pi - abs(vars[0])},
                {'type': 'ineq', 'fun': lambda vars: 2*np.pi - abs(vars[1])},
                {'type': 'ineq', 'fun': lambda vars: 2*np.pi - abs(vars[2])},
            ]
            res = minimize(eq, x0=theta_0, args=(q,), method='SLSQP', constraints=constraints, options=options,tol=1e-15)
            theta_solution_library.append(R.from_quat(q.tolist()[1:4]+[q.tolist()[0]]).as_euler(seq=self.sequence,degrees=False))
            theta_solution.append(res.x)
        return np.array(theta_solution).T
    def plot_trajectory(self):
        plt.subplot(3,1,1)
        plt.plot(self.output_trajectory['traj_pos_1'])
        plt.grid()

        plt.subplot(3,1,2)
        plt.grid()
        plt.plot(self.output_trajectory['traj_pos_2'])
        plt.subplot(3,1,3)

        plt.plot(self.output_trajectory['traj_pos_3'])
        plt.grid()
        plt.show()       
        plt.subplot(3,1,1)
        plt.plot(self.output_trajectory['traj_vel_1'])
        plt.grid()

        plt.subplot(3,1,2)
        plt.grid()
        plt.plot(self.output_trajectory['traj_vel_2'])
        plt.subplot(3,1,3)

        plt.plot(self.output_trajectory['traj_vel_3'])
        plt.grid()
        plt.show()       

        plt.subplot(4,1,1)
        plt.plot(self.output_quat[0,:])
        plt.grid()

        plt.subplot(4,1,2)
        plt.plot(self.output_quat[1,:])
        plt.grid()


        plt.subplot(4,1,3)
        plt.plot(self.output_quat[2,:])
        plt.grid()


        plt.subplot(4,1,4)
        plt.plot(self.output_quat[3,:])
        plt.grid()
        plt.show()


if __name__ == "__main__":
    traj_generator = TrajectoryGenerator(1/8)
    traj_1 = traj_generator.sine_smoooth_velocity(0,[0.5],[0.005],[0.01],[0])
    traj_2 = traj_generator.sine_smoooth_velocity(0,[0.5],[0.005],[0.01],[0])
    traj_3 = traj_generator.sine_smoooth_velocity(0,[0.5],[0.005],[0.01],[0])



    traj_manager = TrajectoryManager("ZYX")
    traj_manager.wrap_trajectories(1,traj_1['pos'],None,None)
    traj_manager.wrap_trajectories(1,None,traj_2['pos'],None)
    traj_manager.wrap_trajectories(1,None,None,traj_3['pos'])    
    traj_manager.wrap_trajectories(0,traj_1['vel'],None,None)
    traj_manager.wrap_trajectories(0,None,traj_2['vel'],None)
    traj_manager.wrap_trajectories(0,None,None,traj_3['vel'])
    
    trajectory_pos_eul = [traj_manager.output_trajectory['traj_pos_1'], traj_manager.output_trajectory['traj_pos_2'],traj_manager.output_trajectory['traj_pos_3']]
    trajectory_pos_eul_dot = [traj_manager.output_trajectory['traj_vel_1'], traj_manager.output_trajectory['traj_vel_2'],traj_manager.output_trajectory['traj_vel_3']]

    plt.subplot(3,1,1)
    plt.plot(traj_manager.output_trajectory['traj_pos_1'])
    plt.grid()

    plt.subplot(3,1,2)
    plt.grid()
    plt.plot(traj_manager.output_trajectory['traj_pos_2'])
    plt.subplot(3,1,3)

    plt.plot(traj_manager.output_trajectory['traj_pos_3'])
    plt.grid()
    plt.show()
    plt.subplot(3,1,1)
    plt.plot(traj_manager.output_trajectory['traj_vel_1'])
    plt.grid()

    plt.subplot(3,1,2)
    plt.grid()
    plt.plot(traj_manager.output_trajectory['traj_vel_2'])
    plt.subplot(3,1,3)

    plt.plot(traj_manager.output_trajectory['traj_vel_3'])
    plt.grid()
    plt.show()
    
    
    
    res = traj_manager.eul_to_omega(trajectory_pos_eul,trajectory_pos_eul_dot)
    plt.subplot(3,1,1)
    plt.plot(res[0,:])
    plt.grid()

    plt.subplot(3,1,2)
    plt.grid()
    plt.plot(res[1,:])
    plt.subplot(3,1,3)

    plt.plot(res[2,:])
    plt.grid()
    plt.show()


  
    # trajectory_quat = traj_manager.eul_to_quat(trajectory_pos_eul)

    # theta_ = traj_manager.quat_to_eul(trajectory_quat)

    # plt.subplot(4,1,1)
    # plt.plot(trajectory_quat[0,:])
    # plt.grid()

    # plt.subplot(4,1,2)
    # plt.plot(trajectory_quat[1,:])
    # plt.grid()


    # plt.subplot(4,1,3)
    # plt.plot(trajectory_quat[2,:])
    # plt.grid()


    # plt.subplot(4,1,4)
    # plt.plot(trajectory_quat[3,:])
    # plt.grid()

    # plt.show()
    # plt.subplot(3,1,1)
    # plt.plot(theta_[0,:],label='optimization')
    # plt.plot(trajectory_pos_eul[0],linestyle='dashed',label='input')
    # plt.grid()

    # plt.subplot(3,1,2)
    # plt.plot(theta_[1,:],label='optimization')
    # plt.plot(trajectory_pos_eul[1],linestyle='dashed',label='input')
    # plt.grid()


    # plt.subplot(3,1,3)
    # plt.plot(theta_[2,:],label='optimization')
    # plt.plot(trajectory_pos_eul[2],linestyle='dashed',label='input')
    # plt.grid()

    # plt.legend()





    # plt.show()
