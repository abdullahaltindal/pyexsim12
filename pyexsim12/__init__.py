"""
Package to create a Python interface for the stochastic ground motion simulation code EXSIM12.
pyexsim12 can be used to:
* Prepare input files for EXSIM12.
* Visualize simulated accelerograms.
* Post-process simulated accelerograms to calculate response spectra and Fourier amplitude spectra.
* Compare simulated accelerograms to ground motion models.
* Store recorded motions (if available) for the simulations, and prepare misfit plots.
"""

import matplotlib.pyplot as plt
from pyexsim12.simulation import Simulation
from pyexsim12.source import Source, SourceSpec, FaultGeom, Hypocenter, Rupture
from pyexsim12.path import Path, TimePads, Crust, GeometricSpreading, QualityFactor, PathDuration
from pyexsim12.misc import Misc
from pyexsim12.amplification import Amplification
from pyexsim12.sites import Sites

plt.interactive(True)
