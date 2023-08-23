from math import isclose

import numpy as np
import pytest
from gemdat.simulation_metrics import SimulationMetrics
from gemdat.volume import trajectory_to_volume, volume_to_structure
from pymatgen.core import Structure


@pytest.fixture
def vasp_vol(vasp_traj):
    trajectory = vasp_traj
    diff_trajectory = trajectory.filter('Li')
    return trajectory_to_volume(trajectory=diff_trajectory, resolution=0.2)


@pytest.vaspxml_available
def test_volume(vasp_vol, vasp_traj):
    data = vasp_vol.data

    n_species = sum(sp.symbol == 'Li' for sp in vasp_traj.species)

    assert isinstance(data, np.ndarray)
    assert data.shape == (99, 49, 49)
    assert data.sum() == n_species * len(vasp_traj)

    # make sure edges are not empty
    s_ = np.s_
    for s in s_[0], s_[:, 0], s_[:, :, 0], s_[-1], s_[:, -1], s_[:, :, -1]:
        assert data[s].sum() != 0


def test_volume_to_structure(vasp_traj, vasp_vol):
    structure = volume_to_structure(vasp_vol, specie='Li')

    assert isinstance(structure, Structure)
    assert len(structure) == 188
    assert np.min(structure.frac_coords) >= 0
    assert np.max(structure.frac_coords) < 1


@pytest.vaspxml_available
def test_tracer(vasp_traj):
    diff_trajectory = vasp_traj.filter('Li')
    metrics = SimulationMetrics(diff_trajectory)

    assert isclose(metrics.particle_density(), 2.4557e28, rel_tol=1e-4)
    assert isclose(metrics.mol_per_liter(), 40.777, rel_tol=1e-4)
    assert isclose(
        metrics.tracer_diffusivity(dimensions=3),
        1.5706e-09,
        rel_tol=1e-4,
    )
    assert isclose(
        metrics.tracer_conductivity(z_ion=1, dimensions=3),
        110.322,
        rel_tol=1e-4,
    )
