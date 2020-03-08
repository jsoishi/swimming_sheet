from scipy.special import erf
import numpy as np

def sheet(x,y,k,omega,t,delta,hw,ampl):
    yy_pos = y + ampl*np.sin(k*x - omega*t)
    yy_vel = y - ampl*omega*np.cos(k*x - omega*t)
    pos = 0.5*(erf((yy_pos+hw)/delta) - erf((yy_pos-hw)/delta))
    vel = 0.5*(erf((yy_vel+hw)/delta) - erf((yy_vel-hw)/delta))
    return pos, vel
