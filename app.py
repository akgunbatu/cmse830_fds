import streamlit as st
import pandas as pd
import numpy as np

# Define the pages
main_page = st.Page("main_page.py", title="Data Introduction")
page_2 = st.Page("page_2.py", title="Missing Values")
page_3 = st.Page("page_3.py", title="Correlation")
page_4 = st.Page("page_4.py", title = "EDA Visualization")

# Set up navigation
pg = st.navigation([main_page, page_2, page_3, page_4])

# Run the selected page
pg.run()