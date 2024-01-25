from __future__ import annotations

from math import isclose

import numpy as np
import pytest
from pymatgen.core import Structure

from gemdat.simulation_metrics import SimulationMetrics
from gemdat.volume import trajectory_to_volume


@pytest.fixture
def vasp_vol(vasp_traj):
    trajectory = vasp_traj
    diff_trajectory = trajectory.filter('Li')
    return trajectory_to_volume(trajectory=diff_trajectory, resolution=0.2)


@pytest.vaspxml_available  # type: ignore
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


# @pytest.mark.skip(reason='https://github.com/GEMDAT-repos/GEMDAT/issues/138')
@pytest.vaspxml_available  # type: ignore
def test_volume_to_structure_centroid(vasp_vol):
    structure = vasp_vol.to_structure(specie='Li', pad=5, method='centroid')

    assert isinstance(structure, Structure)
    assert len(structure) == 157
    assert np.min(structure.frac_coords) >= 0
    assert np.max(structure.frac_coords) < 1
    assert all(sp.symbol == 'Li' for sp in structure.species)


@pytest.vaspxml_available  # type: ignore
def test_volume_to_structure_cluster(vasp_vol):
    structure = vasp_vol.to_structure(specie='Li', pad=5, method='cluster')

    assert hasattr(vasp_vol, 'positions')
    assert hasattr(vasp_vol, 'voxel_mapping')

    assert isinstance(structure, Structure)
    assert len(structure) == 157
    assert np.min(structure.frac_coords) >= 0
    assert np.max(structure.frac_coords) < 1
    assert all(sp.symbol == 'Li' for sp in structure.species)


@pytest.vaspxml_available  # type: ignore
def test_volume_get_free_energy(vasp_vol):
    free_energy = vasp_vol.get_free_energy(temperature=1)
    assert isclose(np.min(free_energy), 0.00061389695902)
    assert isclose(np.average(free_energy[free_energy < 10**5]),
                   0.0008802950738)


@pytest.vaspxml_available  # type: ignore
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


@pytest.vaspxml_available  # type: ignore
def test_vibration_metrics(vasp_traj):
    diff_trajectory = vasp_traj.filter('Li')
    metrics = SimulationMetrics(diff_trajectory)

    speed = metrics.speed()
    assert speed.shape == (48, 3750)
    assert np.allclose(
        speed[::24, ::1000],
        [[0., 0.01348809, -0.02682779, 0.00293911],
         [0., -0.0059066, 0.01617687, 0.01080066]],
    )

    attempt_freq, attempt_freq_std = metrics.attempt_frequency()

    assert np.isclose(attempt_freq, 8496267574320.341)
    assert np.isclose(attempt_freq_std, 857338723421.6494)

    amplitudes = metrics.amplitudes()
    assert amplitudes.shape == (6644, )
    assert np.allclose(
        amplitudes[::1000],
        [
            0.29206348, -0.11529727, 0.33333244, -1.05528599, 0.53086991,
            -0.48080325, 0.61573288
        ],
    )

    assert np.isclose(metrics.vibration_amplitude(), 0.5204299134264091)


@pytest.vaspxml_available  # type: ignore
def test_msd(vasp_traj):
    msd = vasp_traj.mean_squared_displacement()

    assert msd.shape == (104, 3750)
    assert np.isclose(msd[10, -1], 0.03396214706964429)
    assert np.isclose(msd[52, -1], 0.30140820020960746)
    assert np.isclose(msd[85, -1], 11.927261505164097)
