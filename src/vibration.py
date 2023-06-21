import numpy as np


def vibration_properties(traj_coords: np.ndarray, time_step: float):
    """Get the attempt frequency and vibration amplitude."""
    frequency = 1 / time_step

    length = len(traj_coords)


if __name__ == '__main__':
    from gemdat import load_project

    vasp_xml = '/run/media/stef/Scratch/md-analysis-matlab-example/vasprun.xml'

    traj_coords, data = load_project(vasp_xml, diffusing_element='Li')

    # skip first timesteps
    equilibration_steps = 1250

    species = data['species']
    lattice = data['lattice']

    print(species)
    print(lattice)
    print(traj_coords.shape)

    vibration_properties(traj_coords)
