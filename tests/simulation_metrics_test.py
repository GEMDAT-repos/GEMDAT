import numpy as np
from gemdat.simulation_metrics import SimulationMetrics


def test_tracer_metrics(trajectory):
    diff_trajectory = trajectory.filter('B')
    metrics = SimulationMetrics(diff_trajectory)

    assert (np.isclose(metrics.particle_density(), 1e30))
    assert (np.isclose(metrics.mol_per_liter(), 1660.53906))
    assert (np.isclose(metrics.tracer_diffusivity(diffusion_dimensions=3),
                       2.666667e-10))
    assert (np.isclose(
        metrics.tracer_conductivity(z_ion=1, diffusion_dimensions=3),
        4.081278e-09))


def test_vibration_metrics(trajectory):
    diff_trajectory = trajectory.filter('B')
    metrics = SimulationMetrics(diff_trajectory)

    attempt_freq, attempt_freq_std = metrics.attempt_frequency()

    assert np.isclose(attempt_freq, 0.33727)
    assert np.isclose(attempt_freq_std, 0.0)
    assert np.allclose(metrics.amplitudes(), np.array([0.9]))
    assert np.isclose(metrics.vibration_amplitude(), 0.0)
