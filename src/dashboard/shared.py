from pathlib import Path
from tkinter import filedialog

import streamlit as st


def get_data_location(filename='vasprun.xml'):
    data_location = st.session_state.get('data_location', default=filename)

    data_location = st.text_input(f'Location of {filename} on the server',
                                  data_location)

    if st.button(f'Choose location of {filename}'):
        data_location = filedialog.askopenfilename()
        st.session_state.data_location = data_location
        st.experimental_rerun()

    if not data_location or not Path(data_location).exists():
        st.info(f'Choose a `{filename}` file to process')
        st.stop()

    return data_location
