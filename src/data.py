import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel
from pymatgen.core import Structure
from pymatgen.io import vasp


class Data(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    structure: Structure
    trajectory_coords: np.ndarray
    species: object
    lattice: object
    time_step: float

    @classmethod
    def from_cache(cls, cache: Path):
        with open(cache, 'rb') as f:
            data = pickle.load(f)
        return cls(**data)

    def to_cache(self, cache):
        with open(cache, 'wb') as f:
            pickle.dump(self.dict(), f)

    @classmethod
    def from_vasprun(cls, xml_file: Path, cache: Optional[Path] = None):
        if cache and cache.exists():
            return Data.from_cache(cache)
        else:
            run = vasp.Vasprun(
                xml_file,
                parse_dos=False,
                parse_eigen=False,
                parse_projected_eigen=False,
                parse_potcar_file=False,
            )

            structure = run.structures[0]
            trajectory = run.get_trajectory()
            trajectory.to_positions()

            data = {
                'structure': structure,
                'trajectory_coords': trajectory.coords,
                'species': structure.species,
                'lattice': structure.lattice,
                'time_step': run.parameters['POTIM'] * 1e-15,
            }

            # first create object to check pydantic before caching
            ret = cls(**data)

            if cache:
                ret.to_cache(cache)

            return ret
