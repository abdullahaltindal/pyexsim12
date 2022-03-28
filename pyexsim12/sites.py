class Sites:
    """
    Coordinates of sites where the ground motions will be simulated.
    """

    def __init__(self, coords, site_coord_flag=1):
        """
        Args:
            coords: Site coordinates as: [(lat1, lon1), (lat2, lon2) ...].
            site_coord_flag: 1=(lat, long) 2=(R, Az) 3=(N, E)
        """
        self.no_of_sites = len(coords)
        self.coords = coords
        self.site_coord_flag = site_coord_flag

    def __str__(self):
        s = ""
        for i, coord in enumerate(self.coords):
            s += f"Site {i+1}: {coord} \n"
        return s
