"""
Lambert solver — direct Python translation of Curtis Algorithm 5.2
(Orbital Mechanics for Engineering Students, Curtis)
"""

import numpy as np

# ── Stumpff functions ─────────────────────────────────────────────────────────

def stumpff_S(z):
    if   z >  0: return (np.sqrt(z) - np.sin(np.sqrt(z))) / z**1.5
    elif z <  0: return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (-z)**1.5
    else:        return 1.0/6.0

def stumpff_C(z):
    if   z >  0: return (1 - np.cos(np.sqrt(z))) / z
    elif z <  0: return (np.cosh(np.sqrt(-z)) - 1) / (-z)
    else:        return 0.5

# ── Lambert solver (Curtis Algorithm 5.2) ────────────────────────────────────

def lambert(R1, R2, t, string='pro', mu =0.0002959122082322128):
    """
    Solves Lambert's problem (Curtis Algorithm 5.2).

    Parameters
    ----------
    R1, R2  : array-like, shape (3,)   position vectors [AU]
    t       : float                    time of flight   [days]
    string  : 'pro' for prograde, 'retro' for retrograde
    mu      : float                    gravitational parameter [AU^3/day^2]

    Returns
    -------
    V1, V2  : ndarray, shape (3,)      velocity vectors [AU/day]
    """

    R1 = np.array(R1, dtype=float)
    R2 = np.array(R2, dtype=float)

    # Magnitudes
    r1  = np.linalg.norm(R1)
    r2  = np.linalg.norm(R2)
    c12 = np.cross(R1, R2)

    # Transfer angle
    theta = np.arccos(np.clip(np.dot(R1, R2) / (r1 * r2), -1, 1))

    # Prograde / retrograde check 
    if string == 'pro':
        if c12[2] < 0:
            theta = 2*np.pi - theta
    elif string == 'retro':
        if c12[2] >= 0:
            theta = 2*np.pi - theta
    else:
        string = 'pro'
        print('** Prograde trajectory assumed.')

    # Equation 5.35
    A = np.sin(theta) * np.sqrt(r1*r2 / (1 - np.cos(theta)))

    # ── Subfunctions  ────────────────────────

    def y(z):
        """Equation 5.38"""
        return r1 + r2 + A*(z*stumpff_S(z) - 1) / np.sqrt(stumpff_C(z))

    def F(z, dt):
        """Equation 5.40"""
        return (y(z)/stumpff_C(z))**1.5 * stumpff_S(z) + A*np.sqrt(y(z)) - np.sqrt(mu)*dt

    def dFdz(z):
        """Equation 5.43"""
        if z == 0:
            return ( np.sqrt(2)/40 * y(0)**1.5 + A/8 * (np.sqrt(y(0)) + A*np.sqrt(1/(2*y(0)))) )
        else:
            return ( (y(z)/stumpff_C(z))**1.5 * (1/(2*z) * (stumpff_C(z) - 3*stumpff_S(z)/(2*stumpff_C(z))) + 3*stumpff_S(z)**2/(4*stumpff_C(z))) + A/8 * (3*(stumpff_S(z)/stumpff_C(z))*np.sqrt(y(z)) + A*np.sqrt(stumpff_C(z)/y(z))) )

    # ── Find starting z where F changes sign  ────────────────
    # Skip z values where y(z) <= 0
    z = -100.0
    while F(z, t) < 0 if y(z) > 0 else True:
        z += 0.1

    # ── Newton-Raphson iteration (Equation 5.45) ─────────────────────────────
    tol  = 1e-8
    nmax = 5000
    ratio = 1.0
    n     = 0
    while abs(ratio) > tol and n <= nmax:
        n    += 1
        ratio = F(z, t) / dFdz(z)
        z    -= ratio

    if n >= nmax:
        print(f'** Number of iterations exceeds {nmax}')

    # ── Lagrange coefficients (Equations 5.46a,b,d) ──────────────────────────
    f    = 1 - y(z)/r1          # Eq 5.46a
    g    = A * np.sqrt(y(z)/mu) # Eq 5.46b
    gdot = 1 - y(z)/r2          # Eq 5.46d

    # ── Velocities (Equations 5.28, 5.29) ────────────────────────────────────
    V1 = (R2 - f*R1) / g        # Eq 5.28
    V2 = (gdot*R2 - R1) / g     # Eq 5.29

    return V1, V2