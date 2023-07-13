import numpy as np
from pymatgen.core.units import FloatWithUnit
from scipy.constants import Avogadro, Boltzmann, angstrom, elementary_charge


class Tracer:

    @staticmethod
    def calculate_all(data, extras) -> dict[str, float]:
        """Calculate tracer properties.

        Parameters
        ----------
        data : SimulationData
            Input simulation data
        extras : SimpleNamespace
            Extra variables

        Returns
        -------
        extras : dict[str, float]
            Dictionary with calculated parameters
        """
        total_time = extras.total_time

        volume_ang = data.lattice.volume
        volume_m3 = volume_ang * angstrom**3

        particle_density = extras.n_diffusing / volume_m3

        mol_per_liter = (particle_density * 1e-3) / Avogadro

        particle_density = FloatWithUnit(particle_density, 'm^-3')
        mol_per_liter = FloatWithUnit(mol_per_liter, 'mol l^-1')

        print(f'{particle_density=:g} {particle_density.unit}')
        print(f'{mol_per_liter=:g} {mol_per_liter.unit}')

        # Matlab code contains a bug here so I'm not entirely sure what is the definition
        # Matlab code takes the first column, which is equal to 0
        # Do they mean the total displacement (i.e. last column)?
        msd = np.mean(extras.diff_displacements[:, -1]**2)  # Angstrom^2

        temperature = data.temperature

        # Diffusivity = MSD/(2*dimensions*time)
        tracer_diff = (msd * angstrom**2) / (2 * extras.diffusion_dimensions *
                                             total_time)
        # Conductivity = elementary_charge^2 * charge_ion^2 * diffusivity * particle_density / (k_B * T)
        tracer_conduc = ((elementary_charge**2) *
                         (extras.z_ion**2) * tracer_diff *
                         particle_density) / (Boltzmann * temperature)

        tracer_diff = FloatWithUnit(tracer_diff, 'm^2 s^-1')
        tracer_conduc = FloatWithUnit(tracer_conduc, 'S m^-1')

        print(f'{tracer_diff=:g} {tracer_diff.unit}')
        print(f'{tracer_conduc=:g} {tracer_conduc.unit}')

        return {
            'particle_density': particle_density,
            'mol_per_liter': mol_per_liter,
            'tracer_diff': tracer_diff,
            'tracer_conduc': tracer_conduc,
        }
