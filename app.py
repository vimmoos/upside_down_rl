import streamlit as st
from st_pages import add_page_title, get_nav_from_toml

st.set_page_config(layout="wide", page_title="UDRL")

nav = get_nav_from_toml("pages.toml")


pg = st.navigation(nav)

add_page_title(pg)
pg.run()

hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)
