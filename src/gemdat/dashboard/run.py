from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st
from _shared import add_sidebar_logo, get_trajectory_location
from pymatgen.core import Structure

from gemdat import SitesData, __version__, plots
from gemdat.io import get_list_of_known_materials, load_known_material
from gemdat.rdf import radial_distribution
from gemdat.simulation_metrics import SimulationMetrics
from gemdat.trajectory import Trajectory
from gemdat.utils import is_lattice_similar

st.set_page_config(
    page_title='Gemdat dashboard',
    layout='wide',
    page_icon='💎',
    menu_items={
        'Get Help':
        'https://gemdat.readthedocs.io',
        'Report a bug':
        'https://github.com/gemdat-repos/gemdat/issues',
        'About':
        (f'GEMDASH: a dashboard for GEMDAT ({__version__}). '
         '\n\nGEMDAT is Python toolkit for molecular dynamics analysis. '
         '\n\nFor more information, see: https://github.com/GEMDAT-repos/GEMDAT'
         )
    })

add_sidebar_logo()

KNOWN_MATERIALS = get_list_of_known_materials()

with st.sidebar:
    trajectory_location = get_trajectory_location()


@st.cache_data
def _load_trajectory(trajectory_location):
    return Trajectory.from_vasprun(trajectory_location)


with st.spinner('Loading your trajectory data, this might take a while'):
    trajectory = _load_trajectory(trajectory_location)

with st.sidebar:
    # Get list of present elements as tuple of strings
    elements = tuple(set([str(s) for s in trajectory.species]))

    # Set prefered element to lithium if available
    try:
        index = elements.index('Li')
    except ValueError:
        index = 0

    diffusing_element = st.selectbox('Diffusive Element',
                                     elements,
                                     index=index)
    equilibration_steps = int(
        st.number_input('Equilibration Steps',
                        min_value=0,
                        max_value=len(trajectory) - 1,
                        value=1250))

    sites_filename = str(
        st.selectbox('Load sites from known material', KNOWN_MATERIALS))

    st.markdown('Supercell (x,y,z)')
    col1, col2, col3 = st.columns(3)
    supercell = (int(
        col1.number_input('supercell x',
                          min_value=1,
                          value=1,
                          label_visibility='collapsed',
                          help=None)),
                 int(
                     col2.number_input('supercell y',
                                       min_value=1,
                                       value=1,
                                       label_visibility='collapsed')),
                 int(
                     col3.number_input('supercell z',
                                       min_value=1,
                                       value=1,
                                       label_visibility='collapsed')))

    st.markdown('## Radial distribution function')
    do_rdf = st.checkbox('Plot RDFs')

    if do_rdf:
        with st.sidebar:
            max_dist_rdf = st.number_input(
                'Maximum distance for RDF calculation', value=10.0)
            resolution_rdf = st.number_input('Resolution (width of bins)',
                                             value=0.1)

number_of_cols = 3  # Number of figure columns

trajectory = trajectory[equilibration_steps:]
diff_trajectory = trajectory.filter('Li')
metrics = SimulationMetrics(diff_trajectory)

col1, col2, col3 = st.columns(3)

with col1:
    import uncertainties as u
    attempt_freq = u.ufloat(*metrics.attempt_frequency())
    st.metric('Attempt frequency ($\\mathrm{s^{-1}}$)',
              value=f'{attempt_freq:g}')
    st.metric('Vibration amplitude ($\\mathrm{Å}$)',
              value=f'{metrics.vibration_amplitude():g}')
with col2:
    st.metric('Particle density ($\\mathrm{m^{-3}}$)',
              value=f'{metrics.particle_density():g}')
    st.metric('Mol per liter ($\\mathrm{mol/l}$)',
              value=f'{metrics.mol_per_liter():g}')
with col3:
    st.metric('Tracer diffusivity ($\\mathrm{m^2/s}$)',
              value=f'{metrics.tracer_diffusivity(dimensions=3):g}')
    st.metric('Tracer conductivity ($\\mathrm{S/m}$)',
              value=f'{metrics.tracer_conductivity(z_ion=1, dimensions=3):g}')

tab1, tab2, tab3 = st.tabs(['Default plots', 'RDF plots', 'Density plots'])

sites_structure = load_known_material(sites_filename, supercell=supercell)

if not is_lattice_similar(trajectory.get_lattice(), sites_structure):
    st.error('Lattices are not similar!')
    st.text(f'{sites_filename}: {sites_structure.lattice.parameters}')
    st.text(
        f'{trajectory_location.name}: {trajectory.get_lattice().parameters}')
    st.stop()


def _structure_hash_func(obj: Structure) -> dict:
    return obj.as_dict()


def _trajectory_hash_func(obj: Trajectory) -> np.ndarray:
    return obj.positions


@st.cache_data(hash_funcs={
    Structure: _structure_hash_func,
    Trajectory: _trajectory_hash_func
})
def _SitesData(**kwargs):
    return SitesData(**kwargs)


with tab1:
    st.title('Trajectory and jumps')

    with st.spinner('Calculating jumps...'):
        sites = _SitesData(
            structure=sites_structure,
            trajectory=trajectory,
            floating_specie=diffusing_element,
            n_parts=10,
        )

    diff_trajectory = trajectory.filter(diffusing_element)

    figures = (
        plots.displacement_per_element(trajectory=trajectory),
        plots.displacement_per_site(trajectory=diff_trajectory),
        plots.displacement_histogram(trajectory=diff_trajectory),
        plots.frequency_vs_occurence(trajectory=diff_trajectory),
        plots.vibrational_amplitudes(trajectory=diff_trajectory),
        plots.jumps_vs_distance(sites=sites),
        plots.jumps_vs_time(sites=sites),
        plots.collective_jumps(sites=sites),
        plots.jumps_3d(sites=sites),
    )

    # automagically divide the plots over the number of columns
    for num, col in enumerate(st.columns(number_of_cols)):
        for figure in figures[num::number_of_cols]:
            col.pyplot(figure)


def _sites_hash_func(obj: SitesData) -> tuple[Any, Any, Any]:
    return obj.structure.frac_coords, obj.trajectory.positions, obj.floating_specie


@st.cache_data(hash_funcs={SitesData: _sites_hash_func})
def _radial_distribution(**kwargs):
    return radial_distribution(**kwargs)


if do_rdf:
    with tab2:
        st.title('Radial distribution function')

        with st.spinner('Calculating RDFs...'):
            rdf_data = _radial_distribution(
                sites=sites,
                max_dist=max_dist_rdf,
                resolution=resolution_rdf,
            )

        rdf_figures = [
            plots.radial_distribution(rdfs) for rdfs in rdf_data.values()
        ]

        # automagically divide the plots over the number of columns
        for num, col in enumerate(st.columns(number_of_cols)):
            for figure in rdf_figures[num::number_of_cols]:
                col.pyplot(figure)

with tab3:
    from gemdat.volume import trajectory_to_volume

    st.title('Density plots')

    with st.form('density form'):
        density_resolution = st.number_input('Density resolution (Å)',
                                             min_value=0.1,
                                             value=0.3,
                                             step=0.05)

        density_submit = st.form_submit_button('Generate density')

    if density_submit:
        vol = trajectory_to_volume(
            trajectory=diff_trajectory,
            resolution=density_resolution,
        )

        structure = vol.to_structure()
        chart = plots.density(vol, structure)
        st.plotly_chart(chart)
