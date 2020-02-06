import numpy as np
from scipy.special import erf

def mask(x,y,phi,delta):
    cos, sin = np.cos(phi), np.sin(phi)
    x1,x2 = (x*cos+y*sin),(y*cos-x*sin)
    return 0.5*(1-erf(C(x1,x2)/delta))

def normal(x,y,phi,delta):
    cos, sin = np.cos(phi), np.sin(phi)
    x1,x2 = (x*cos+y*sin),(y*cos-x*sin)
    M,Mx,My = C(x1,x2,grad=True)
    M = np.exp(-(M/delta)**2)/(delta*np.sqrt(np.pi))
    return M*(cos*Mx-sin*My), M*(sin*Mx+cos*My)

def C(x,y,grad=False):
    Z00,Z20,Z02 = -1, 0.25, 1
    Z = Z20*x**2 + Z02*y**2 + Z00
    if grad:
        return Z, 2*Z20*x, 2*Z02*y
    else:
        return Z

def B(x,y,grad=False):
    Z00 = -(999/6400)
    Z10 =  (1629/4000)
    Z01 =  (1011/250)
    Z20 = -(2391/10000)
    Z11 =  (449/1250)
    Z02 =  (21297/5000)
    Z30 = -(34/625)
    Z21 = -(18/25)
    Z12 = -(849/1250)
    Z03 = -(104/625)
    Z40 =  (16/625)
    Z31 =  (32/625)
    Z22 =  (24/625)
    Z13 =  (8/625)
    Z04 =  (1/625)
    Z = Z00 + Z10*x + Z01*y + Z20*(x**2) + Z11*x*y + Z02*(y**2) + Z30*(x**3) \
        + Z21*(x**2)*y + Z12*x*(y**2) + Z03*(y**3) + Z40*(x**4) + Z31*(x**3)*y \
        + Z22*(x*y)**2 + Z13*x*(y**3) + Z04*(y**4)
    if grad:
        Zx = Z10 + 2*Z20*x + Z11*y + 3*Z30*(x**2) + 2*Z21*x*y \
            + Z12*(y**2) + 4*Z40*(x**2) + 3*Z31*(x**2)*y + 2*Z22*x*(y**2) + Z13*(y**3)
        Zy = Z01 + Z11*x + 2*Z02*y + Z21*(x**2) + 2*Z12*x*y \
            + 3*Z03*(y**2) + Z31*(x**3) + 2*Z22*x*(y**2) + 3*Z13*x*(y**2) + 4*Z04*(y**3)
        return Z, Zx, Zy
    else:
        return Z










