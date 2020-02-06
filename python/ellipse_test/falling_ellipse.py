"""
Dedalus script for 2D incompressible hydrodynamics with moving immersed boundary.

This script uses a Fourier basis in both y directions.

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge.py` script in this
folder can be used to merge distributed analysis sets from parallel runs,
and the `plot_2d_series.py` script can be used to plot the snapshots.

To run using 4 processes (e.g), you could use:
    $ mpiexec -n 4 python3 falling_ellipse.py


"""

import numpy as np
import body as bdy
from mpi4py import MPI
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.extras import plot_tools

import logging
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD
rank = comm.rank

# Parameters
dt      =  0.001
Ly,Lz   =  8,8.
ν,γ,δ   =  0.1,40,0.1
μ,I     =  1.2, 5/4 # density ratio, moment of inertia
gravity =  1

# Initial body parameters
y0,V0 = 0,0
z0,W0 = 0,0
φ0,ω0 = np.pi/6,0

# Create bases and domain
y_basis = de.Fourier('y',384, interval=(-Ly, Ly), dealias=3/2)
z_basis = de.Fourier('z',384, interval=(-Lz, Lz), dealias=3/2)
domain  = de.Domain([y_basis, z_basis], grid_dtype=np.float64)

# Setup mask function
y,z = domain.grids(scales=domain.dealias)
K,V,W = domain.new_field(),domain.new_field(),domain.new_field()
K.set_scales(domain.dealias,keep_data=False)
V.set_scales(domain.dealias,keep_data=False)
W.set_scales(domain.dealias,keep_data=False)
Δy,Δz = np.mod(y-y0+Ly,2*Ly)-Ly, np.mod(z-z0+Lz,2*Lz)-Lz
K['g'], V['g'], W['g'] =  bdy.mask(Δy,Δz,φ0,δ), V0 - ω0*Δz, W0 + ω0*Δy

# 2D Incompressible hydrodynamics
problem = de.IVP(domain, variables=['p','v','w','ζ'])

problem.parameters['ν']   = ν
problem.parameters['γ']   = γ
problem.parameters['K']   = K
problem.parameters['V']   = V
problem.parameters['W']   = W
problem.add_equation("dt(v) + ν*dz(ζ) + dy(p) =  ζ*w -γ*K*(v-V)")
problem.add_equation("dt(w) - ν*dy(ζ) + dz(p) = -ζ*v -γ*K*(w-W)")
problem.add_equation("ζ + dz(v) - dy(w) = 0")
problem.add_equation("dy(v) + dz(w) = 0",condition="(ny != 0) or (nz != 0)")
problem.add_equation("p = 0",condition="(ny == 0) and (nz == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# Initial conditions
ζ = solver.state['ζ']
v = solver.state['v']

# Integration parameters
solver.stop_sim_time = np.inf
solver.stop_wall_time = 10*60*60.
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots_test', iter=20, max_writes=50)
snapshots.add_task("p")
snapshots.add_task("v")
snapshots.add_task("w")
snapshots.add_task("ζ")

# Runtime monitoring properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=20)
flow.add_property("abs(ζ)", name='q')

# Tasks for force computation
force = flow_tools.GlobalFlowProperty(solver, cadence=1)
force.add_property("K*(v-V)", name='F0')
force.add_property("K*(w-W)", name='G0')
force.add_property("-z*K*(v-V)+y*K*(w-W)", name='T0')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        solver.step(dt)
        F0 = γ*force.volume_average('F0')
        G0 = γ*force.volume_average('G0')
        τ0 = γ*force.volume_average('T0') + z0*F0 - y0*G0
        F0,G0,τ0 = F0/μ, G0/μ - gravity, τ0/(μ*I)
        y0 = y0 + V0*dt
        z0 = z0 + W0*dt
        V0 = V0 + F0*dt
        W0 = W0 + G0*dt
        φ0 = φ0 + ω0*dt
        ω0 = ω0 + τ0*dt
        Δy = np.mod(y-y0+Ly,2*Ly)-Ly
        Δz = np.mod(z-z0+Lz,2*Lz)-Lz
        K['g'] =  bdy.mask(Δy,Δz,φ0,δ)
        V['g'] =  V0 - ω0*Δz
        W['g'] =  W0 + ω0*Δy
        if (solver.iteration-1) % 20 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max ζ = %f' %flow.max('q'))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))






