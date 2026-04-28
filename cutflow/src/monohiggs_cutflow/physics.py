import math

def dphi(phi1: float, phi2: float) -> float:
    """Delta-phi wrapped to [0, pi]."""
    x = abs(phi1 - phi2)
    while x > math.pi:
        x = abs(x - 2.0 * math.pi)
    return x

def dr(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    """Delta-R using eta/phi."""
    return math.sqrt((eta1 - eta2) ** 2 + dphi(phi1, phi2) ** 2)

def passes_minmax(val, vmin, vmax) -> bool:
    if vmin is not None and val < vmin:
        return False
    if vmax is not None and val > vmax:
        return False
    return True
