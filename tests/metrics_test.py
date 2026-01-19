from __future__ import annotations

import numpy as np

from gemdat.metrics import TrajectoryMetrics, TrajectoryMetricsStd, ArrheniusFit


def test_tracer_metrics(trajectory):
    diff_trajectory = trajectory.filter('B')
    metrics = TrajectoryMetrics(diff_trajectory)

    assert np.isclose(metrics.particle_density(), 1e30)
    assert np.isclose(metrics.mol_per_liter(), 1660.53906)
    assert np.isclose(metrics.tracer_diffusivity(dimensions=3), 2.666667e-10)
    assert np.isclose(metrics.tracer_conductivity(z_ion=1, dimensions=3), 4.081278e-09)


def test_tracer_metrics_std(trajectory):
    diff_trajectory = trajectory.filter('B')
    metrics = TrajectoryMetricsStd(diff_trajectory.split(2, equal_parts=True))

    td = metrics.tracer_diffusivity(dimensions=3)
    assert np.isclose(td.n, np.array([3.334e-23]))
    assert np.isclose(td.s, np.array([2.77555756e-17]))

    tc = metrics.tracer_conductivity(z_ion=1, dimensions=3)
    assert np.isclose(tc.n, np.array([05.038e-10]))
    assert np.isclose(tc.s, np.array([2.77555756e-17]))


def test_vibration_metrics(trajectory):
    diff_trajectory = trajectory.filter('B')
    metrics = TrajectoryMetrics(diff_trajectory)

    assert np.allclose(metrics.speed(), [[0.0, 0.2, 0.2, 0.2, 0.3]])

    attempt_freq, attempt_freq_std = metrics.attempt_frequency()

    assert np.isclose(attempt_freq, 0.33727)
    assert np.isclose(attempt_freq_std, 0.0)
    assert np.allclose(metrics.amplitudes(), np.array([0.9]))
    assert np.isclose(metrics.vibration_amplitude(), 0.0)


def test_vibration_metrics_std(trajectory):
    diff_trajectory = trajectory.filter('B')
    metrics = TrajectoryMetricsStd(diff_trajectory.split(2, equal_parts=True))

    speed = metrics.speed()
    assert np.allclose(speed[0], [0.0, 0.2])

    amplitudes_mean, amplitudes_std = metrics.amplitudes()
    assert np.isclose(amplitudes_mean, np.array([0.2]))
    assert np.isclose(amplitudes_std, np.array([2.77555756e-17]))


def test_arrhenius(trajectory_list):
    arrhenius = ArrheniusFit.from_trajectories(trajectories=trajectory_list, diffusing_specie='B', n_parts=2)

    assert np.isclose(arrhenius.particle_density(), 1e30)
    assert np.allclose(arrhenius.diffusivities, np.array([2.2e-27, 1.6e-26, 2.9e-26, 6.1e-27]))

    ea = arrhenius.activation_energy()
    assert np.isclose(ea.n, np.array([0.072302]))
    assert np.isclose(ea.s, np.array([0.0400955]))

    prefactor = arrhenius.prefactor()
    assert np.isclose(prefactor.n, np.array([2.07e-26]))
    assert np.isclose(prefactor.s, np.array([1.57e-26]))

    diffusivity = arrhenius.extrapolate_diffusivity(temperature=100)
    assert np.isclose(diffusivity.n, np.array([4.7e-30]))
    assert np.isclose(diffusivity.s, np.array([1.8e-29]))

    conductivity = arrhenius.extrapolate_conductivity(temperature=100, z_ion=1)
    assert np.isclose(conductivity.n, np.array([8.7e-17]))
    assert np.isclose(conductivity.s, np.array([3.4e-16]))
