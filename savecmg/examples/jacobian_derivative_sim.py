import os
import sys
import numpy as np

sys.path.append(os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir))

from modules.cmga import ControlMomentGyroAssembly
import sympy as sym

CMGA = ControlMomentGyroAssembly([0,0,0,np.pi/2],[True,True,True,True])
CMGA.initialize_cmgs_array(cmgs_momenta=[10,10,10,10])
dJ_dtheta_symbolic = CMGA.differentiate_symbolic_jacobian()
dJ_dtheta = CMGA.substitute_to_symbolic_jacobian(np.array([0,2,0,0]))

print(np.array(dJ_dtheta))