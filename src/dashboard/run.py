import streamlit as st
from _shared import add_sidebar_logo, get_trajectory_location
from gemdat import SitesData, __version__, plots
from gemdat.calculate import calculate_all
from gemdat.io import get_list_of_known_materials, load_known_material
from gemdat.rdf import calculate_rdfs, plot_rdf
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
    equilibration_steps = st.number_input('Equilibration Steps',
                                          min_value=0,
                                          max_value=len(trajectory) - 1,
                                          value=1250)

    sites_filename = st.selectbox('Load sites from known material',
                                  KNOWN_MATERIALS)

    st.markdown('Supercell (x,y,z)')
    col1, col2, col3 = st.columns(3)
    supercell = (col1.number_input('supercell x',
                                   min_value=1,
                                   value=1,
                                   label_visibility='collapsed',
                                   help=None),
                 col2.number_input('supercell y',
                                   min_value=1,
                                   value=1,
                                   label_visibility='collapsed'),
                 col3.number_input('supercell z',
                                   min_value=1,
                                   value=1,
                                   label_visibility='collapsed'))

number_of_cols = 3  # Number of figure columns

with st.spinner('Processing trajectory...'):
    trajectory = trajectory[equilibration_steps:]
    extras = calculate_all(trajectory, diffusing_element=diffusing_element)

col1, col2, col3 = st.columns(3)

with col1:
    import uncertainties as u
    attempt_freq = u.ufloat(extras.attempt_freq, extras.attempt_freq_std)
    st.metric('Attempt frequency ($\\mathrm{s^{-1}}$)',
              value=f'{attempt_freq:g}')
    st.metric('Vibration amplitude ($\\mathrm{Å}$)',
              value=f'{extras.vibration_amplitude:g}')
with col2:
    st.metric('Particle density ($\\mathrm{m^{-3}}$)',
              value=f'{extras.particle_density:g}')
    st.metric('Mol per liter ($\\mathrm{mol/l}$)',
              value=f'{extras.mol_per_liter:g}')
with col3:
    st.metric('Tracer diffusivity ($\\mathrm{m^2/s}$)',
              value=f'{extras.tracer_diff:g}')
    st.metric('Tracer conductivity ($\\mathrm{S/m}$)',
              value=f'{extras.tracer_conduc:g}')

tab1, tab2 = st.tabs(['Default plots', 'RDF plots'])

sites_structure = load_known_material(sites_filename, supercell=supercell)

with tab1:
    st.title('GEMDAT pregenerated figures')

    if not is_lattice_similar(trajectory.get_lattice(), sites_structure):
        st.error('Lattices are not similar!')
        st.text(f'{sites_filename}: {sites_structure.lattice.parameters}')
        st.text(
            f'{trajectory_location.name}: {trajectory.get_lattice().parameters}'
        )
        st.stop()

    with st.spinner('Calculating jumps...'):
        sites = SitesData(sites_structure)
        sites.calculate_all(trajectory=trajectory, extras=extras)

    diff_trajectory = trajectory.filter(diffusing_element)

    figures = (
        plots.plot_displacement_per_element(trajectory=trajectory),
        plots.plot_displacement_per_site(trajectory=diff_trajectory),
        plots.plot_displacement_histogram(trajectory=diff_trajectory),
        plots.plot_frequency_vs_occurence(trajectory=trajectory,
                                          sites=sites,
                                          **vars(extras)),
        plots.plot_vibrational_amplitudes(trajectory=trajectory,
                                          sites=sites,
                                          **vars(extras)),
        plots.plot_jumps_vs_distance(trajectory=trajectory, sites=sites),
        plots.plot_jumps_vs_time(trajectory=trajectory, sites=sites),
        plots.plot_collective_jumps(trajectory=trajectory, sites=sites),
        plots.plot_jumps_3d(trajectory=trajectory, sites=sites),
    )

    # automagically divide the plots over the number of columns
    for num, col in enumerate(st.columns(number_of_cols)):
        for figure in figures[num::number_of_cols]:
            col.pyplot(figure)

with tab2:
    st.title('GEMDAT RDF plots')

    do_rdf = st.checkbox(
        'Enable RDF plots (This might increase the page load time significantly)'
    )
    if do_rdf:
        with st.sidebar:
            st.markdown('RDF specific parameters')
            max_dist_rdf = st.number_input(
                'Maximum Distance for RDF calculation', value=10.0)
            resolution_rdf = st.number_input(
                'Resolution for RDF calculation (width of bins)', value=0.1)

        with st.spinner('Calculating RDFs...'):
            rdfs = calculate_rdfs(
                trajectory=trajectory,
                sites=sites,
                species=diffusing_element,
                max_dist=max_dist_rdf,
                resolution=resolution_rdf,
            )
        rdf_figures = [
            plot_rdf(rdf, name=state) for state, rdf in rdfs.items()
        ]

        # automagically divide the plots over the number of columns
        for num, col in enumerate(st.columns(number_of_cols)):
            for figure in rdf_figures[num::number_of_cols]:
                col.pyplot(figure)
