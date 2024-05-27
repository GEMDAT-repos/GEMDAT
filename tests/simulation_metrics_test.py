from __future__ import annotations

import numpy as np

from gemdat.simulation_metrics import SimulationMetrics, SimulationMetricsStd


def test_tracer_metrics(trajectory):
    diff_trajectory = trajectory.filter('B')
    metrics = SimulationMetrics(diff_trajectory)

    assert np.isclose(metrics.particle_density(), 1e30)
    assert np.isclose(metrics.mol_per_liter(), 1660.53906)
    assert np.isclose(metrics.tracer_diffusivity(dimensions=3), 2.666667e-10)
    assert np.isclose(metrics.tracer_conductivity(z_ion=1, dimensions=3), 4.081278e-09)


def test_tracer_metrics_std(trajectory):
    diff_trajectory = trajectory.filter('B')
    metrics = SimulationMetricsStd(diff_trajectory.split(2, equal_parts=True))

    td = metrics.tracer_diffusivity(dimensions=3)
    assert np.isclose(td.n, np.array([3.334e-23]))
    assert np.isclose(td.s, np.array([2.77555756e-17]))

    tc = metrics.tracer_conductivity(z_ion=1, dimensions=3)
    assert np.isclose(tc.n, np.array([05.038e-10]))
    assert np.isclose(tc.s, np.array([2.77555756e-17]))


def test_vibration_metrics(trajectory):
    diff_trajectory = trajectory.filter('B')
    metrics = SimulationMetrics(diff_trajectory)

    assert np.allclose(metrics.speed(), [[0.0, 0.2, 0.2, 0.2, 0.3]])

    attempt_freq, attempt_freq_std = metrics.attempt_frequency()

    assert np.isclose(attempt_freq, 0.33727)
    assert np.isclose(attempt_freq_std, 0.0)
    assert np.allclose(metrics.amplitudes(), np.array([0.9]))
    assert np.isclose(metrics.vibration_amplitude(), 0.0)


def test_vibration_metrics_std(trajectory):
    diff_trajectory = trajectory.filter('B')
    metrics = SimulationMetricsStd(diff_trajectory.split(2, equal_parts=True))

    speed = metrics.speed()
    assert np.allclose(speed[0], [0.0, 0.2])

    amplitudes_mean, amplitudes_std = metrics.amplitudes()
    assert np.isclose(amplitudes_mean, np.array([0.2]))
    assert np.isclose(amplitudes_std, np.array([2.77555756e-17]))
