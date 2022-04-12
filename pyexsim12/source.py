from dataclasses import dataclass


class Source:
    """
    A Source object contains all the simulation parameters on source properties.
    """

    def __init__(self, source_spec, fault_geom, hypo, rupture):
        """
        Args:
            source_spec (SourceSpec): Source spectrum information. Expects a parameter of SourceSpec class.
            fault_geom(FaultGeom): Fault geometry information. Expects a parameter of FaultGeom class.
            hypo(Hypocenter): Hypocenter location. Expects a parameter of Hypocenter class.
            rupture(Rupture): Rupture information. Expects a parameter of Rupture class.
        """
        self.source_spec = source_spec
        self.fault_geom = fault_geom
        self.hypo = hypo
        self.rupture = rupture

    def __str__(self):
        mw = self.source_spec.mw
        stress_drop = self.source_spec.stress_drop
        kappa = self.source_spec.kappa
        strike, dip, depth = self.fault_geom.angles
        return f"Mw: {mw}\nStress drop: {stress_drop} bars\nkappa: {kappa} s\n" \
               f"Strike: {strike}°\nDip: {dip}°\nDepth: {depth} km"


@dataclass()
class SourceSpec:
    """
    Contains parameters for source spectrum.
    Attributes:
        mw: Moment magnitude
        stress_drop: Stress drop in bars
        kappa: Kappa value in seconds. If kappa_flag != 1 -> f_max in Hz.
        kappa_flag: f_max or kappa for high frequency attenuation.(0=f_max; 1=kappa)
    """
    mw: float
    stress_drop: float
    kappa: float
    kappa_flag: int = 1

    def __iter__(self):
        return iter([self.mw, self.stress_drop, self.kappa_flag, self.kappa])


class FaultGeom:
    """
    A FaultGeom object contains information on fault geometry, such as location of the upper edge of the fault,
    dip & strike angles, depth, faulting type, length and width of the fault and mesh size in the direction of
    length and width of the fault.
    """

    def __init__(self, fault_edge, angles, fault_type, len_width):
        """
        Args:
            fault_edge: Latitude and longitude of upper edge of the fault in the format: (lat, lon)
            angles: [strike, dip, depth]
            fault_type: (S=strikeslip; R=reverse; N=normal; U=undifferentiated)
            len_width: [length, width, d_length, d_width, stress_ref]. Default value for stress_ref is 70 bars.
        """
        # Assign a stress_ref value of 70 bars if not entered:
        if len(len_width) == 4:
            len_width.append(70)

        self.fault_edge = fault_edge
        self.angles = angles
        self.fault_type = fault_type
        self.len_width = len_width


@dataclass()
class Hypocenter:
    """
    Hypocenter location in along fault and down dip distance from the fault. (-1.0, -1.0 for a random location).
    number of iterations over hypocenter is only used if random location is selected
    Attributes:
        hypo_along_fault: Hypocenter location in along fault. -1 for random.
        hypo_down_dip:  Hypocenter location in down dip distance. -1 for random.
        iters: Number of iterations. Only used for random location.
    """
    hypo_along_fault: float
    hypo_down_dip: float
    iters: int = 1
    # def __init__(self, hypo_along_fault, hypo_down_dip, iters=1):
    #     self.hypo_along_fault = hypo_along_fault
    #     self.hypo_down_dip = hypo_down_dip
    #     self.iters = iters

    def __iter__(self):
        return iter([self.hypo_along_fault, self.hypo_down_dip, self.iters])


@dataclass()
class Rupture:
    """
    Fault rupture parameters
        Attributes:
        vrup_beta: Rupture velocity / Beta
        risetime: Type of risetime. (1=original, 2=1/f0). Default is 1.
        i_slip_weight: -1: Unity slip for all subfaults
                        0: Specify slips from the text file
                        1: Random weights.
                        Default is 0.
        slip_weights: Filename of the slip weights. Default is "slip_weights.txt".
    """
    vrup_beta: float
    risetime: int = 1
    i_slip_weight: int = 0
    slip_weights: str = "slip_weights.txt"
