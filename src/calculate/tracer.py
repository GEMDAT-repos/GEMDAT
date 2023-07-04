import numpy as np
from gemdat.constants import avogadro, e_charge, k_boltzmann


class Tracer:

    @staticmethod
    def calculate_all(
        data,
        displacements: np.ndarray,
        equilibration_steps: int,
        diffusing_element: str,
        dimensions: int = 3,
        z_ion: int = 1,
        **kwargs,
    ) -> dict[str, float]:
        """Calculate tracer properties.

        Parameters
        ----------
        data : SimulationData
            Input simulation data
        equilibration_steps : int
            Number of equilbiration steps
        diffusing_element : str
            Name of the diffusing element
        displacements : np.ndarray
            Input array with displacements
        dimensions : int
            Number of diffusion dimensions
        z_ion : int
            Ionic charge of the diffusing ion

        Returns
        -------
        extras : dict[str, float]
            Dictionary with calculate parameters
        """
        time_step = data.time_step
        n_diffusing = sum([e.name == diffusing_element for e in data.species])

        angstrom_to_meter = 1e-10

        n_steps = len(data.trajectory_coords) - equilibration_steps
        len(data.species)

        total_time = n_steps * time_step

        volume_ang = data.lattice.volume
        volume_m3 = volume_ang * angstrom_to_meter**3

        particle_density = n_diffusing / volume_m3

        mol_per_liter = (particle_density * 1e-3) / avogadro

        print(f'{particle_density=:g} m^-3')
        print(f'{mol_per_liter=:g} mol/l')

        # grab displacements for diffusing element only
        idx = np.argwhere([e.name == diffusing_element for e in data.species])
        diff_displacements = displacements[idx].squeeze()

        # Matlab code contains a bug here so I'm not entirely sure what is the definition
        # Matlab code takes the first column, which is equal to 0
        # Do they mean the total displacement (i.e. last column)?
        msd = np.mean(diff_displacements[:, -1]**2)  # Angstron^2

        temperature = data.temperature

        # Diffusivity = MSD/(2*dimensions*time)
        tracer_diff = (msd * angstrom_to_meter**2) / (2 * dimensions *
                                                      total_time)
        # Conductivity = elementary_charge^2 * charge_ion^2 * diffusivity * particle_density / (k_B * T)
        tracer_conduc = ((e_charge**2) * (z_ion**2) * tracer_diff *
                         particle_density) / (k_boltzmann * temperature)

        print(f'{tracer_diff=:g} m^2/s')
        print(f'{tracer_conduc=:g} S/m')

        return {
            'particle_density': particle_density,
            'mol_per_liter': mol_per_liter,
            'tracer_diff': tracer_diff,
            'tracer_conduc': tracer_conduc,
        }
