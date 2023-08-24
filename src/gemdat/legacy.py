from gemdat import SitesData, load_known_material, plots
from gemdat.rdf import radial_distribution
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
    calc_rdfs: bool = False,
    rdf_res: float = 0.1,
    rdf_max_dist: int = 10,
    start_end: tuple[int, int] = (5000, 7500),
    nr_steps_frame: int = 5,
) -> tuple[Trajectory, SitesData]:
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
    calc_rdfs : bool, optional
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

    diff_trajectory = trajectory.filter(diff_elem)

    sites_structure = load_known_material(material, supercell=supercell)

    sites = SitesData(
        structure=sites_structure,
        trajectory=trajectory,
        floating_specie=diff_elem,
        n_parts=nr_parts,
    )

    plots.displacement_per_element(trajectory=trajectory)
    plots.displacement_per_site(trajectory=diff_trajectory)
    plots.displacement_histogram(trajectory=diff_trajectory)
    plots.frequency_vs_occurence(trajectory=diff_trajectory)
    plots.vibrational_amplitudes(trajectory=diff_trajectory)
    plots.jumps_vs_distance(trajectory=trajectory,
                            sites=sites,
                            jump_res=jump_res)
    plots.jumps_vs_time(trajectory=trajectory, sites=sites)
    plots.collective_jumps(trajectory=trajectory, sites=sites)
    plots.jumps_3d(trajectory=trajectory, sites=sites)

    plots.jumps_3d_animation(
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

    trajectory_to_vasp_volume(trajectory=trajectory.filter(diff_elem),
                              structure=trajectory.get_structure(0),
                              resolution=density_resolution,
                              filename=filename)

    if calc_rdfs:
        rdf_data = radial_distribution(
            sites=sites,
            max_dist=rdf_max_dist,
            resolution=rdf_res,
        )
        for rdfs in rdf_data.values():
            plots.radial_distribution(rdfs.values())

    return trajectory, sites