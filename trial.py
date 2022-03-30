import gmm
from pyexsim12 import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

source_spec = SourceSpec(7.1, 100, 0.047)
fault_geom = FaultGeom((40.77, 31.45), [264.0, 64.0, 0.0], "S", [65.0, 25.0, 5.0, 5.0, 70.0])
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

sim.create_input_file(save=True)
sim.run(override=True)
duzce_recorded = pd.read_csv("duzce_recorded.txt", names=["EW", "NS", "V"], delim_whitespace=True)
recorded_ew = np.array(duzce_recorded["EW"])
recorded_ns = np.array(duzce_recorded["NS"])
sim.rec_motions = (1, "EW", recorded_ew, 0.005)
sim.rec_motions = (1, "NS", recorded_ns, 0.005)


sim.plot_rec_fas(1, "EW")
plt.xlim(left=0.1, right=50)

# slip = np.array([[0.4, 0.6, 1.2, 1.35, 1.05, 1.2, 0.8, 1.2, 1.8, 1.7, 1.3, 0.9, 0.6],
#                  [0.4, 0.9, 1.3, 1.4, 2.0, 1.8, 1.1, 1.8, 2.7, 2.6, 1.9, 1.3, 0.75],
#                  [0.28, 0.55, 0.9, 1.2, 1.5, 1.7, 1.2, 2.1, 2.6, 2.4, 1.8, 1.2, 0.75],
#                  [0.1, 0.25, 0.6, 0.7, 1.05, 1.35, 1.5, 1.95, 2.1, 1.6, 1.05, 0.75, 0.6],
#                  [0.1, 0.1, 0.3, 0.45, 0.5, 0.7, 1.0, 1.2, 1.2, 0.6, 0.3, 0.2, 0.1]
#                  ])

slip = np.array([[0.745, 0.801, 0.986],
                 [0.097, 0.561, 0.406]])
# np.savetxt("filename.txt", slip, fmt="%1.3f", delimiter="\t")
sim.create_slip_file(slip, "filename.txt")