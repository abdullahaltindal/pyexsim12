import gmm
from pyexsim12 import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

source_spec = SourceSpec(7.1, 100, 0.047)
fault_geom = FaultGeom((40.77, 31.45), [264.0, 64.0, 5.0], "S", [65.0, 25.0, 5.0, 5.0, 70.0])
hypo = Hypocenter(7, 3)
rupture = Rupture(0.8, i_slip_weight=0, slip_weights="slip_weights.txt")
source = Source(source_spec, fault_geom, hypo, rupture)

time_pads = TimePads(50.0, 20.0, 0.005)
crust = Crust(3.7, 2.8)
geometric_spreading = GeometricSpreading(3, [(1.0, -1.0), (30.0, -0.6), (50.0, -0.5)])
quality_factor = QualityFactor(0.0, 88, 0.9)
path_duration = PathDuration()
path = Path(time_pads, crust, geometric_spreading, quality_factor, path_duration)

amplification = Amplification("site_amps_dzc.txt", crustal_amp="crustal_dzc.txt",
                              empirical_amp="empirical_amps.txt")
misc = Misc()
sites = Sites([(40.85, 31.17)])

sim = Simulation(source, path, amplification, misc, sites)

# sim.create_input_file(save=True)
sim.run()

print(sim.get_rjb(1))