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

# Source 2
source_spec2 = SourceSpec(7.1, 100, 0.04)
source_spec3 = SourceSpec(7.1, 100, 0.03)
source2 = Source(source_spec2, fault_geom, hypo, rupture)
source3 = Source(source_spec3, fault_geom, hypo, rupture)
sim2 = Simulation(source2, path, amplification, Misc(), sites)
sim3 = Simulation(source3, path, amplification, Misc(), sites)

sim.create_input_file(save=True)
sim2.create_input_file(save=True)
sim3.create_input_file(save=True)


sim.run()
sim2.run()
sim3.run()
#%%
fig, axs = plt.subplots()
sim.plot_rp(site=1, axis=axs, plot_dict=dict(label="$\kappa=0.047$"))
sim2.plot_rp(site=1, axis=axs, plot_dict=dict(label="$\kappa=0.04$"))
sim3.plot_rp(site=1, axis=axs, plot_dict=dict(label="$\kappa=0.03$"))
axs.legend()
axs.set_xscale("log")
axs.set_yscale("log")
axs.set_xlim(right=4)


# sim.create_input_file(save=True)
# sim.run(override=True)
# duzce_recorded = pd.read_csv("duzce_recorded.txt", names=["EW", "NS", "V"], delim_whitespace=True)
# recorded_ew = np.array(duzce_recorded["EW"])
# recorded_ns = np.array(duzce_recorded["NS"])
# sim.rec_motions = (1, "EW", recorded_ew, 0.005)
# sim.rec_motions = (1, "NS", recorded_ns, 0.005)
#
# sim.plot_acc(1)
