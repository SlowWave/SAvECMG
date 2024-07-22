import numpy as np
import matplotlib.pyplot as plt



class TrajectoryGenerator():
    def __init__(self,pos_0,sample_time):
        self.pos_0 = pos_0
        self.sample_time = sample_time
        self.trajectory = {
            "time" : list(),
            "pos"  : list(),
            "vel"  : list(),
            "acc"  : list(),            
        }
        
    def step_position(self,pos_fin,stop_time):

        #Define the number of trajectory 
        number_trajectory = len(pos_fin)
        time_count = 0
        pos_vec = [self.pos_0]
        pos_vec += pos_fin 
        
        for i in range(number_trajectory):
            #Estrapolate the numebr of samples
            n_samples = round(stop_time[i]/self.sample_time)
            for j in range(n_samples):
                self.trajectory["time"].append(time_count*self.sample_time)
                self.trajectory["pos"].append(pos_vec[i])
                self.trajectory["vel"].append(0)
                self.trajectory["acc"].append(0)
                time_count += 1
        self.trajectory["time"].append(time_count*self.sample_time)
        self.trajectory["pos"].append(pos_vec[i+1])
        self.trajectory["vel"].append(0)
        self.trajectory["acc"].append(0)

        return self.trajectory
    
    
    
    def linear_position(self,pos_fin,vel,stop_time):

        #Define the number of trajectory 
        number_trajectory = len(pos_fin)
        time_count = 0
        pos_prec = self.pos_0
        time_prec = 0
        for i in range(number_trajectory):
            #Estrapolate the number of samples
            delta_pos = pos_fin[i] - pos_prec
            n_samples_trajectory = round(abs(delta_pos/vel[i])/self.sample_time ) 
            n_samples_wait = round(stop_time[i]/self.sample_time)
            vel_sign = np.sign(delta_pos)
            for k in range(n_samples_wait):
                self.trajectory["time"].append(time_count*self.sample_time)
                self.trajectory["vel"].append(0)
                self.trajectory["pos"].append(pos_prec)
                self.trajectory["acc"].append(0)
                time_count += 1
            time_prec = self.trajectory["time"][-1]
            
            for j in range(n_samples_trajectory):
                self.trajectory["time"].append(time_count*self.sample_time)
                self.trajectory["vel"].append(vel_sign*abs(vel[i]))
                self.trajectory["pos"].append(pos_prec + vel_sign*abs(vel[i])*(self.trajectory["time"][-1]-time_prec))
                self.trajectory["acc"].append(0)
                time_count += 1
                
            pos_prec = self.trajectory["pos"][-1]
        return self.trajectory


    def trapezoidal_velocity(self,pos_fin,vel,acc_max,stop_time):
        number_trajectory = len(pos_fin)
        # Initialize variables
        pos_prec = self.pos_0
        time_prec = 0
        vel_prec = 0
        time = 0
        t_1 = 0
        pos_1 = 0
        vel_1 = 0
        t_2 = 0
        pos_2 = 0
        vel_2 = 0
        for i in range(number_trajectory):
            self.trajectory["time"].append(time_prec)
            self.trajectory["vel"].append(0)
            self.trajectory["pos"].append(pos_prec)
            self.trajectory["acc"].append(0)
            acc = acc_max[i]
            delta_pos = pos_fin[i] - pos_prec
            vel_sign = np.sign(delta_pos)


            if abs(delta_pos) < abs(acc*(vel[i] / acc)**2):
                vel[i] =  vel_sign * np.sqrt(abs(acc*delta_pos))
            # Compute the acceleration interval
            t_acc1 = abs(vel[i] / acc)
            # Compute the space during the acceleration
            s_acc = acc * t_acc1**2
            # Compute the space and the time during the constant velocity 
            s_vel = delta_pos - vel_sign*s_acc


            t_vel = abs(s_vel/vel[i])
            # Compute the instant of the starting of deceleration
            t_acc2 = t_acc1 + t_vel    
            t_tot = t_acc1 + t_acc2  

            n_samples_wait = round(stop_time[i]/self.sample_time)

            n_sample_trajectory = round(t_tot/self.sample_time)+1
            # Allocate samples to waiting time
            
            for _ in range(n_samples_wait):
                self.trajectory["time"].append(time)
                self.trajectory["vel"].append(0)
                self.trajectory["pos"].append(pos_prec)
                self.trajectory["acc"].append(0)
                time_prec = time
                time += self.sample_time

            # Build trajectory path
            for _ in range(n_sample_trajectory):
                if  time - time_prec <= t_acc1:
                    delta_time = time-time_prec
                    self.trajectory["acc"].append(vel_sign*acc )
                    self.trajectory["vel"].append(vel_prec + vel_sign*acc*delta_time)
                    self.trajectory["pos"].append(pos_prec + vel_prec*delta_time + vel_sign*1/2*acc*delta_time**2)
                    pos_1 = self.trajectory["pos"][-1]
                    vel_1 = self.trajectory["vel"][-1]
                    t_1 = time
                    t_2 = time
                    pos_2 = self.trajectory["pos"][-1]
                    vel_2 = self.trajectory["vel"][-1]
                elif time - time_prec <= t_acc2:
                    delta_time = time-t_1
                    self.trajectory["acc"].append(0)
                    self.trajectory["vel"].append(vel_1)
                    self.trajectory["pos"].append(pos_1 + vel_1*delta_time )
                    pos_2 = self.trajectory["pos"][-1]
                    vel_2 = self.trajectory["vel"][-1]
                    t_2 = time
                else:
                    
                    delta_time = time - t_2
                    self.trajectory["acc"].append(-vel_sign*acc)
                    self.trajectory["vel"].append(vel_2 - vel_sign*acc*delta_time )
                    self.trajectory["pos"].append(pos_2 + vel_2 * delta_time - vel_sign*1/2*acc*delta_time**2 )
                self.trajectory["time"].append(time)
                time += self.sample_time



            pos_prec = self.trajectory["pos"][-1]
            vel_prec = self.trajectory["vel"][-1]
            time_prec = self.trajectory["time"][-1]    
        return self.trajectory
        
    
    
    
   
    def sine_smoooth_velocity(self,pos_fin,vel,acc_max,stop_time):
        number_trajectory = len(pos_fin)
        # Initialize variables
        time_count = 0
        pos_prec = 0
        time_prec = 0
        vel_prec = 0
        time = 0
        t_1 = 0
        pos_1 = 0
        vel_1 = 0
        t_2 = 0
        pos_2 = 0
        vel_2 = 0


        for i in range(number_trajectory):
            acc = acc_max[i] / 2
            delta_pos = pos_fin[i] - pos_prec
            vel_sign = np.sign(delta_pos)
            if abs(delta_pos) < abs(acc*(vel[i] / acc)**2):
                vel[i] =  vel_sign * np.sqrt(abs(acc*delta_pos))
            #Estrapolate the number of samples

            # Compute the acceleration interval
            t_acc1 = abs(vel[i] / acc)
            # Compute the space during the acceleration
            s_acc = acc * t_acc1**2
            # Compute the space and the time during the constant velocity 
            s_vel = delta_pos - vel_sign*s_acc

            t_vel = abs(s_vel/vel[i])
            # Compute the instant of the starting of deceleration
            t_acc2 = t_acc1 + t_vel    
            t_tot = 2*t_acc1 + t_vel  
            n_samples_wait = round(stop_time[i]/self.sample_time)
            
            omega_acc = 2*np.pi/t_acc1

            # Allocate samples to waiting time
            for _ in range(n_samples_wait):
                self.trajectory["time"].append(time_count*self.sample_time)
                self.trajectory["vel"].append(0)
                self.trajectory["pos"].append(pos_prec)
                self.trajectory["acc"].append(0)
                time_count += 1
                time_prec = self.trajectory["time"][-1]
                time = time_prec
            # Build trajectory path
            while time - time_prec < t_tot:
                self.trajectory["time"].append(time_count*self.sample_time)
                if  self.trajectory["time"][-1] - time_prec <= t_acc1:
                    delta_time = self.trajectory["time"][-1]-time_prec
                    self.trajectory["acc"].append(vel_sign*acc - vel_sign*acc*np.cos(omega_acc*delta_time))
                    self.trajectory["vel"].append(vel_prec + vel_sign*acc*delta_time - vel_sign*acc/omega_acc*np.sin(omega_acc*delta_time))
                    self.trajectory["pos"].append(pos_prec + vel_prec*delta_time + vel_sign*1/2*acc*delta_time**2 + vel_sign*acc/omega_acc**2*(np.cos(omega_acc*delta_time)-1))
                    pos_1 = self.trajectory["pos"][-1]
                    vel_1 = self.trajectory["vel"][-1]
                    t_1 = self.trajectory["time"][-1]
                    t_2 = self.trajectory["time"][-1]
                    pos_2 = self.trajectory["pos"][-1]
                    vel_2 = self.trajectory["vel"][-1]
                elif  t_acc1 < self.trajectory["time"][-1] - time_prec <= t_acc2:
                    delta_time = self.trajectory["time"][-1]-t_1
                    self.trajectory["acc"].append(0)
                    self.trajectory["vel"].append(vel_1)
                    self.trajectory["pos"].append(pos_1 + vel_1*delta_time )
                    pos_2 = self.trajectory["pos"][-1]
                    vel_2 = self.trajectory["vel"][-1]
                    t_2 = self.trajectory["time"][-1]

                else:
                    delta_time = self.trajectory["time"][-1] - t_2
                    self.trajectory["acc"].append(-vel_sign*acc+vel_sign*acc*np.cos(omega_acc*delta_time))
                    self.trajectory["vel"].append(vel_2 - vel_sign*acc*delta_time + vel_sign*acc/omega_acc*np.sin(omega_acc*delta_time))
                    self.trajectory["pos"].append(pos_2 + vel_2 * delta_time - vel_sign*1/2*acc*delta_time**2 - vel_sign*acc/omega_acc**2*(np.cos(omega_acc*delta_time)-1))
                time_count += 1
                time = self.trajectory["time"][-1]

            pos_prec = self.trajectory["pos"][-1]
            vel_prec = self.trajectory["vel"][-1]
            time_prec = self.trajectory["time"][-1]    
        return self.trajectory
        
            
                
                
        
if __name__== "__main__":
    stop_time = [0,0]
    ref = [0.2]   
    vel = [2,2]
    acc = [0.1,0.1]
    generator = TrajectoryGenerator(0,1/32)  
    trajectory = generator.trapezoidal_velocity(ref,vel,acc,stop_time)
    
    plt.subplot(3,1,1)
    plt.plot(trajectory['time'],trajectory['pos'])   
    plt.grid()

    plt.subplot(3,1,2)
    plt.plot(trajectory['time'],trajectory['vel'])   
    plt.grid()
    plt.subplot(3,1,3)
    plt.plot(trajectory['time'],trajectory['acc'])   
    plt.grid()
    plt.show()