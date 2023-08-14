from types import SimpleNamespace

import numpy as np
from gemdat.calculate.tracer import Tracer


def test_vibration_calculate_all(trajectory):
    extras = SimpleNamespace(
        total_time=1e-12,
        n_diffusing=1,
        diffusion_dimensions=3,
        z_ion=1,
    )
    diff_trajectory = trajectory.filter('B')

    ret = Tracer.calculate_all(diff_trajectory, extras)

    assert (np.isclose(ret['particle_density'], 1e30))
    assert (np.isclose(ret['mol_per_liter'], 1660.53906))
    assert (np.isclose(ret['tracer_diff'], 2.666667e-10))
    assert (np.isclose(ret['tracer_conduc'], 20406.389030))
