import streamlit as st
from _shared import add_sidebar_logo, get_data_location
from gemdat import SimulationData, __version__, plot_all

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

with st.sidebar:
    data_location = get_data_location(filename='vasprun.xml')

with st.spinner('Loading your data, this might take a while'):
    data = SimulationData.from_vasprun(data_location)

with st.sidebar:
    # Get list of present elements as tuple of strings
    elements = tuple(set([str(s) for s in data.species]))

    # Set prefered element to lithium if available
    try:
        index = elements.index('Li')
    except ValueError:
        index = 0

    diffusing_element = st.selectbox('Diffusive Element',
                                     elements,
                                     index=index)
    equilibration_steps = st.number_input(
        'Equilibration Steps',
        min_value=0,
        max_value=len(data.trajectory_coords) - 1,
        value=1250,
        step=100)
    number_of_cols = int(
        st.number_input('Number of figure columns',
                        min_value=1,
                        max_value=10,
                        value=3))

extra = data.calculate_all(equilibration_steps=equilibration_steps,
                           diffusing_element=diffusing_element)

col1, col2, col3 = st.columns(3)

with col1:
    import uncertainties as u
    attempt_freq = u.ufloat(extra.attempt_freq, extra.attempt_freq_std)
    st.metric('Attempt frequency ($\\mathrm{s^{-1}}$)',
              value=f'{attempt_freq:g}')
    st.metric('Vibration amplitude ($\\mathrm{Å}$)',
              value=f'{extra.vibration_amplitude:g}')
with col2:
    st.metric('Particle density ($\\mathrm{m^{-3}}$)',
              value=f'{extra.particle_density:g}')
    st.metric('Mol per liter ($\\mathrm{mol/l}$)',
              value=f'{extra.mol_per_liter:g}')
with col3:
    st.metric('Tracer diffusivity ($\\mathrm{m^2/s}$)',
              value=f'{extra.tracer_diff:g}')
    st.metric('Tracer conductivity ($\\mathrm{S/m}$)',
              value=f'{extra.tracer_conduc:g}')

st.title('GEMDAT pregenerated figures')

figures = plot_all(data=data, **vars(extra), show=False)

# automagically divide the plots over the number of columns
for num, col in enumerate(st.columns(number_of_cols)):
    for figure in figures[num::number_of_cols]:
        col.pyplot(figure)
