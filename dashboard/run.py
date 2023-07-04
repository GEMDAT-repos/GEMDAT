from pathlib import Path

import pygwalker as pyg
import streamlit as st
import streamlit.components.v1 as components
import xarray as xr
from gemdat import Data, plot_all

data = Data.from_vasprun(Path('../example/vasprun.xml'), cache=Path('cache'))
extra = data.calculate_all(equilibration_steps=1250, diffusing_element='Li')

st.set_page_config(page_title='Gemdat gemdash dashboard', layout='wide')

fig_tab, pyg_tab = st.tabs(['Figures', 'PyGWalker'])

with fig_tab:
    # Add Title
    st.title('GEMDAT pregenerated figures')
    figures = plot_all(data=data, **extra, show=False)
    col1, col2 = st.columns(2)
    with col1:
        for figure in figures[::2]:
            st.pyplot(figure)
    with col2:
        for figure in figures[1::2]:
            st.pyplot(figure)

with pyg_tab:

    # Add Title
    st.title('Use Pygwalker for GEMDAT')

    # Import your data
    data_xr = xr.DataArray(extra['speed'],
                           dims=['element', 'time']).isel(time=slice(0, 1000))
    data_df = data_xr.to_dataframe(name='speed')

    # Generate the HTML using Pygwalker
    pyg_html = pyg.walk(data_df, return_html=True)

    # Embed the HTML into the Streamlit app
    components.html(pyg_html, height=1000, scrolling=True)
