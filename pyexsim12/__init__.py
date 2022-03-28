import matplotlib.pyplot as plt
# from pathlib import Path
# import os
from pyexsim12.simulation import Simulation
from pyexsim12.source import Source, SourceSpec, FaultGeom, Hypocenter, Rupture
from pyexsim12.path import Path, TimePads, Crust, GeometricSpreading, QualityFactor, PathDuration
from pyexsim12.misc import Misc
from pyexsim12.amplification import Amplification
from pyexsim12.sites import Sites

plt.interactive(True)
