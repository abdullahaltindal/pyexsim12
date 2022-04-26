import gmm
from pyexsim12 import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %% Creating the Source object
src_spec = SourceSpec(mw=7.1, stress_drop=100, kappa=0.047)  # kappa_flag = 1 by default, so no need to change anything
fault_geom = FaultGeom(fault_edge=(40.77, 31.45),
                       angles=[264.0, 64.0, 0.0],
                       fault_type="S",
                       len_width=[65.0, 25.0, 5.0, 5.0, 70.0])
hypo = Hypocenter(hypo_along_fault=7, hypo_down_dip=3)
# Create the slip file
# First we will create a matrix of slip weights. Since we have 65/5=13 subfaults along the length, and 25/5=5 along the
# width, dimensions of the array will be 5 x 13.
slip_matrix = np.array([[0.4, 0.6, 1.2, 1.35, 1.05, 1.2, 0.8, 1.2, 1.8, 1.7, 1.3, 0.9, 0.6],
                        [0.4, 0.9, 1.3, 1.4, 2.0, 1.8, 1.1, 1.8, 2.7, 2.6, 1.9, 1.3, 0.75],
                        [0.28, 0.55, 0.9, 1.2, 1.5, 1.7, 1.2, 2.1, 2.6, 2.4, 1.8, 1.2, 0.75],
                        [0.1, 0.25, 0.6, 0.7, 1.05, 1.35, 1.5, 1.95, 2.1, 1.6, 1.05, 0.75, 0.6],
                        [0.1, 0.1, 0.3, 0.45, 0.5, 0.7, 1.0, 1.2, 1.2, 0.6, 0.3, 0.2, 0.1]])
# Now we will pass this array into the create_slip_file method of simulation module. Since exsim is located at the
# folder "exsim12", which is the default argument for the exsim_folder parameter, we will leave it as default.
simulation.create_slip_file(slip_matrix=slip_matrix, filename="slip_weights.txt")
rupture = Rupture(vrup_beta=0.8, slip_weights="slip_weights.txt")
src = Source(src_spec, fault_geom, hypo, rupture)

time_pads = TimePads(tpad1=50.0, tpad2=20.0, delta_t=0.005)
crust = Crust(beta=3.7, rho=2.8)
geom_spread = GeometricSpreading(n_seg=3, spread=[(1.0, -1.0), (30.0, -0.6), (50.0, -0.5)])
q_factor = QualityFactor(0.0, 88, 0.9)
path_dur = PathDuration()  # No input is provided as the default values will be used
path = Path(time_pads, crust, geom_spread, q_factor, path_dur)

# %% Creating amplification files and Amplification object
simulation.create_amp(freq=[0.1953, 0.9766, 5.859, 8.887, 11.72],
                      amp=[5.115, 7.155, 0.7477, 0.5308, 0.2812],
                      filename="site_amp_tutorial.txt",
                      header="Site amplification file for pyexsim12 tutorial")

simulation.create_amp(freq=[0.01, 0.1, 0.2, 0.3, 0.5, 0.9, 1.25, 1.8, 3.0, 5.3, 8.0, 14.0],
                      amp=[1.0, 1.02, 1.03, 1.05, 1.07, 1.09, 1.11, 1.12, 1.13, 1.14, 1.15, 1.15],
                      filename="crustal_amp_tutorial.txt",
                      header="Crustal amplification file for pyexsim12 tutorial")

amp = Amplification(site_amp="site_amp_tutorial.txt", crustal_amp="crustal_amp_tutorial.txt")

# %% Create Misc and Sites objects

misc = Misc()  # Everything will be kept as default
sites = Sites(
    [(2.5, 90), (2.5, 180), (2.5, 264), (2.5, 0)],
    site_coord_flag=2)

sim = Simulation(src, path, amp, misc, sites)
sim.create_input_file(save=True)  # Create the input file for EXSIM12
sim.run(override=True)  # Run the simulation, if output files for this configuration exist, they will be overwritten
