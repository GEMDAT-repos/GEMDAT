import numpy as np
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

        print(f'{particle_density=:g} m^-3')
        print(f'{mol_per_liter=:g} mol/l')

        # Matlab code contains a bug here so I'm not entirely sure what is the definition
        # Matlab code takes the first column, which is equal to 0
        # Do they mean the total displacement (i.e. last column)?
        msd = np.mean(extras.diff_displacements[:, -1]**2)  # Angstron^2

        temperature = data.temperature

        # Diffusivity = MSD/(2*dimensions*time)
        tracer_diff = (msd * angstrom**2) / (2 * extras.diffusion_dimensions *
                                             total_time)
        # Conductivity = elementary_charge^2 * charge_ion^2 * diffusivity * particle_density / (k_B * T)
        tracer_conduc = ((elementary_charge**2) *
                         (extras.z_ion**2) * tracer_diff *
                         particle_density) / (Boltzmann * temperature)

        print(f'{tracer_diff=:g} m^2/s')
        print(f'{tracer_conduc=:g} S/m')

        return {
            'particle_density': particle_density,
            'mol_per_liter': mol_per_liter,
            'tracer_diff': tracer_diff,
            'tracer_conduc': tracer_conduc,
        }
