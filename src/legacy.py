from types import SimpleNamespace

from gemdat import SitesData, load_known_material, plot_all
from gemdat.calculate import calculate_all
from gemdat.plots.jumps import plot_jumps_3d_animation
from gemdat.rdf import calculate_rdfs, plot_rdf
from gemdat.trajectory import Trajectory
from gemdat.volume import trajectory_to_vasp_volume


def analyse_md(
    vasp_xml: str,
    *,
    diff_elem: str,
    material: str,
    supercell: tuple[int, int, int] | None = None,
    equil_time: float = 2.5E-12,
    diffusion_dimensions: int = 3,
    z_ion: int = 1,
    nr_parts: int = 10,
    dist_collective: float = 4.5,
    density_resolution: float = 0.2,
    jump_res: float = 0.1,
    rdfs: bool = False,
    rdf_res: float = 0.1,
    rdf_max_dist: int = 10,
    start_end: tuple[int, int] = (5000, 7500),
    nr_steps_frame: int = 5,
) -> tuple[Trajectory, SitesData, SimpleNamespace]:
    """Analyse md data.

    Parameters
    ----------
    vasp_xml : str
        Simulation data input file
    diff_elem : str
        Name of diffusing element
    material : str
        Name of known material with sites for diffusing elements
    supercell : tuple[int, int, int], optional
        Super cell to apply to the known material
    equil_time : float, optional
        Equilibration time in seconds (1e-12 = 1 picosecond)
    diffusion_dimensions : int, optional
        Number of diffusion dimensions
    z_ion : int, optional
        Ionic charge of diffusing ion
    nr_parts : int, optional
        In how many parts to divide your simulation for statistics
    dist_collective : float, optional
        Maximum distance for collective motions in Angstrom
    density_resolution : float, optional
        Resolution for the diffusing atom density plot in Angstrom
    jump_res: float, optional
        Resolution for the number of jumps vs distance plot in Angstrom
    rdfs : bool, optional
        Calculate and show radial distribution functions
    rdf_res : float, optional
        Resolution of the rdf bins in Angstrom
    rdf_max_dist : int, optional
        Maximal distanbe of the rdf in Angstrom
    start_end : tuple[int, int], optional
        The time steps for which to start and end the movie
    nr_steps_frame : int, optional
        How many time steps per frame in the movie, increase to get shorter and faster movies

    Returns
    -------
    tuple[SimulationData, SitesData, SimpleNamespace]
        Output data
    """
    trajectory = Trajectory.from_vasprun(vasp_xml)

    equilibration_steps = round(equil_time / trajectory.time_step)

    trajectory = trajectory[equilibration_steps:]

    extras = calculate_all(
        trajectory,
        diffusing_element=diff_elem,
        z_ion=z_ion,
        diffusion_dimensions=diffusion_dimensions,
        n_parts=nr_parts,
    )

    sites_structure = load_known_material(material, supercell=supercell)

    sites = SitesData(sites_structure)
    sites.calculate_all(trajectory=trajectory,
                        extras=extras,
                        dist_collective=dist_collective)

    plot_all(trajectory=trajectory,
             sites=sites,
             **vars(extras),
             show=False,
             jump_res=jump_res)

    plot_jumps_3d_animation(
        trajectory=trajectory,
        sites=sites,
        t_start=start_end[0],
        t_stop=start_end[1],
        skip=nr_steps_frame,
        decay=0.05,
        interval=20,
    )

    filename = 'volume.vasp'
    print(f'Writing trajectory as a volume to `{filename}')

    trajectory_to_vasp_volume(coords=extras.diff_coords,
                              structure=trajectory.get_structure(0),
                              resolution=density_resolution,
                              filename=filename)

    if rdfs:
        rdf_data = calculate_rdfs(
            trajectory=trajectory,
            sites=sites,
            diff_coords=extras.diff_coords,
            n_steps=extras.n_steps,
            max_dist=rdf_max_dist,
            resolution=rdf_res,
        )
        for name, rdf in rdf_data.items():
            plot_rdf(rdf, name=name)

    return trajectory, sites, extras