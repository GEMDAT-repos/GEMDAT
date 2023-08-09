from types import SimpleNamespace

import numpy as np
from gemdat.calculate.tracer import Tracer


def test_vibration_calculate_all(trajectory):
    extras = SimpleNamespace(
        diff_displacements=np.array([[0., 0.1, 0.2, 0.3, 0.4]]),
        total_time=1e-12,
        n_diffusing=1,
        diffusion_dimensions=3,
        z_ion=1,
    )

    ret = Tracer.calculate_all(trajectory, extras)

    assert (np.isclose(ret['particle_density'], 1e30))
    assert (np.isclose(ret['mol_per_liter'], 1660.5390671738464))
    assert (np.isclose(ret['tracer_diff'], 2.6666666666666673e-10))
    assert (np.isclose(ret['tracer_conduc'], 4030.8916603094012))
