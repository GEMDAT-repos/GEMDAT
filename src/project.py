from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from pymatgen.core import Element, Lattice
from pymatgen.io.vasp.outputs import Vasprun

PATH_COORDS = Path('traj_coords.npy')
PATH_DATA = Path('data.yaml')
VASP_XML = 'vasprun.xml'


def load_project(directory: str | Path,
                 diffusing_element: Optional[str] = None):
    """Read coordinates and data from vasp_xml."""
    vasp_xml = Path(directory) / VASP_XML

    if not (PATH_COORDS.exists() and PATH_DATA.exists()):
        assert vasp_xml.exists()

        vasprun = Vasprun(
            vasp_xml,
            parse_dos=False,
            parse_eigen=False,
            parse_projected_eigen=False,
            parse_potcar_file=False,
        )

        traj = vasprun.get_trajectory()

        structure = vasprun.structures[0]

        data = {
            'species': [e.as_dict() for e in structure.species],
            'lattice':
            structure.lattice.as_dict(),
            'parameters':
            vasprun.parameters,
            # Size of the timestep (*1E-15 = in femtoseconds)
            'time_step':
            vasprun.time_step['POTIM'] * 1e-15,
            # Temperature of the MD simulation
            'temperature':
            vasprun.parameters['TEBEG'],
            # Number of diffusing elements
            'nr_diffusing':
            sum([
                e.name == diffusing_element
                for e in vasprun.initial_structure.species
            ])
        }

        with open('data.yaml', 'w') as f:
            yaml.dump(data, f)

        traj.to_positions()
        np.save('traj_coords.npy', traj.coords)

        traj_coords = traj.coords
    else:
        traj_coords = np.load('traj_coords.npy')

        with open('data.yaml') as f:
            data = yaml.unsafe_load(f)

    data['species'] = [Element.from_dict(e) for e in data['species']]
    data['lattice'] = Lattice.from_dict(data['lattice'])
    data['diffusing_element'] = diffusing_element

    return traj_coords, data
