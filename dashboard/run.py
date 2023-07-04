from pathlib import Path

import streamlit as st
from gemdat import SimulationData, plot_all

#import pygwalker as pyg
#import streamlit.components.v1 as components

st.set_page_config(page_title='Gemdat gemdash dashboard', layout='wide')

fig_tab, pyg_tab = st.tabs(['Figures', 'PyGWalker'])

with st.sidebar:
    data_location = st.text_input('Location of data', 'vasprun.xml')
    cache_location = st.text_input('Location of cache', 'cache')

data = SimulationData.from_vasprun(Path(data_location),
                                   cache=Path(cache_location))

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
    number_of_cols = st.number_input('Number of figure columns',
                                     min_value=1,
                                     max_value=10,
                                     value=3)

extra = data.calculate_all(equilibration_steps=equilibration_steps,
                           diffusing_element=diffusing_element)

with fig_tab:
    st.title('GEMDAT pregenerated figures')

    figures = plot_all(data=data, **extra, show=False)

    # automagically divide the plots over the number of columns
    for num, col in enumerate(st.columns(number_of_cols)):
        for figure in figures[num::number_of_cols]:
            col.pyplot(figure)

with pyg_tab:
    pass
#    extra = data.calculate_all(equilibration_steps=1250, diffusing_element=diffusing_element)
#
#    # Add Title
#    st.title('Use Pygwalker for GEMDAT')
#
#    # Import your data
#    data_xr = xr.DataArray(extra['speed'],
#                           dims=['element', 'time']).isel(time=slice(0, 1000))
#    data_df = data_xr.to_dataframe(name='speed')
#
#    # Generate the HTML using Pygwalker
#    pyg_html = pyg.walk(data_df, return_html=True)
#
#    # Embed the HTML into the Streamlit app
#    components.html(pyg_html, height=1000, scrolling=True)
