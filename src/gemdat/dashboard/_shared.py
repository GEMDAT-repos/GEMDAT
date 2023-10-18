from __future__ import annotations

import base64
import os
from importlib.resources import files
from pathlib import Path
from tkinter import filedialog

import streamlit as st

data_directory = Path(files('gemdat') / 'data')  # type: ignore


def get_trajectory_location(filename='vasprun.xml'):
    if trajectory_location := os.environ.get('VASP_XML'):
        st.info(f'Got `{trajectory_location}` via environment variable.')
        return Path(trajectory_location)

    trajectory_location = st.session_state.get('trajectory_location',
                                               default=filename)

    st.markdown('Select input trajectory')
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        trajectory_location = st.text_input('filename',
                                            trajectory_location,
                                            label_visibility='collapsed')
    with col2:
        if st.button('Browse'):
            trajectory_location = filedialog.askopenfilename()
            st.session_state.trajectory_location = trajectory_location
            st.experimental_rerun()

    if not trajectory_location:
        st.info(f'Select `{filename}` to continue')
        st.stop()

    trajectory_location = Path(trajectory_location).expanduser()

    if not trajectory_location.exists():
        st.info(
            f'Could not find `{trajectory_location}`, select `{filename}` to continue'
        )
        st.stop()

    return trajectory_location


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
