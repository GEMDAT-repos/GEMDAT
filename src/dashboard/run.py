from pathlib import Path
from tkinter import filedialog

import streamlit as st
from gemdat import SimulationData, plot_all

st.set_page_config(page_title='Gemdat gemdash dashboard', layout='wide')

fig_tab, _ = st.tabs(['Figures', 'Other Tabs'])

data_location = st.session_state.get('data_location', default='vasprun.xml')

with st.sidebar:
    data_location = st.text_input('Location of vasprun.xml on the server',
                                  data_location)
    if st.button('Choose location of vasprun.xml'):
        data_location = filedialog.askopenfilename()
        st.session_state.data_location = data_location
        st.experimental_rerun()

if not data_location or not Path(data_location).exists():
    st.info('choose a Vasprun xml file to process')
    st.stop()

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

with fig_tab:
    st.title('GEMDAT pregenerated figures')

    figures = plot_all(data=data, **vars(extra), show=False)

    # automagically divide the plots over the number of columns
    for num, col in enumerate(st.columns(number_of_cols)):
        for figure in figures[num::number_of_cols]:
            col.pyplot(figure)
