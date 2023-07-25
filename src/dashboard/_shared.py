import base64
import os
from importlib.resources import files
from pathlib import Path
from tkinter import filedialog

import streamlit as st

data_directory = Path(files('gemdat') / 'data')  # type: ignore


def get_data_location(filename='vasprun.xml'):
    if data_location := os.environ.get('VASP_XML'):
        st.info(f'Got `{data_location}` via environment variable.')
        return Path(data_location)

    data_location = st.session_state.get('data_location', default=filename)

    data_location = st.text_input(f'Select input `{filename}`', data_location)

    if st.button('Browse...'):
        data_location = filedialog.askopenfilename()
        st.session_state.data_location = data_location
        st.experimental_rerun()

    if not data_location:
        st.info(f'Select `{filename}` to continue')
        st.stop()

    data_location = Path(data_location).expanduser()

    if not data_location.exists():
        st.info(
            f'Could not find `{data_location}`, select `{filename}` to continue'
        )
        st.stop()

    return data_location


@st.cache_data
def get_base64_of_bin_file(png_file):
    with open(png_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def build_markup_for_logo(png_file,
                          background_position='50% 2%',
                          margin_top='0%',
                          image_width='60%',
                          image_height='',
                          anchor='stSidebar'
                          # anchor='stSidebarNav'  # Anchor to navigation
                          ):
    binary_string = get_base64_of_bin_file(png_file)
    return f"""
            <style>
                [data-testid="{anchor}"] {{
                    background-image: url("data:image/png;base64,{binary_string}");
                    background-repeat: no-repeat;
                    background-position: {background_position};
                    margin-top: {margin_top};
                    background-size: {image_width} {image_height};
                }}
            </style>
            """


def add_sidebar_logo():
    """Based on: https://stackoverflow.com/a/73278825."""
    png_file = data_directory / 'logo.png'
    logo_markup = build_markup_for_logo(png_file)
    st.markdown(
        logo_markup,
        unsafe_allow_html=True,
    )
