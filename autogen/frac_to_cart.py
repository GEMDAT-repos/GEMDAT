import numpy as np


def frac_to_cart(frac_pos, lattice):
    # Calculate cartesian coordinates from the fractional position
    cart_pos = np.zeros(3)
    for i in range(3):
        cart_pos[i] = lattice[0, i] * frac_pos[0] + lattice[
            1, i] * frac_pos[1] + lattice[2, i] * frac_pos[2]
    return cart_pos
