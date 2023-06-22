import numpy as np


def calc_dist_sqrd_frac(frac1, frac2, lattice):
    # Given two fractional coordinates and the lattice, calculate dist^2
    frac = frac1 - frac2

    # Periodic Boundary Conditions
    for i in range(3):
        if frac[i] > 0.5:
            frac[i] = frac[i] - 1.0
        elif frac[i] < -0.5:
            frac[i] = frac[i] + 1.0

    # Convert to Cartesian coordinates
    cart = np.dot(frac, lattice)

    # Calculate distance^2
    dist_sqrd = cart[0]**2 + cart[1]**2 + cart[2]**2

    return dist_sqrd
