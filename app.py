import streamlit as st
import helper


st.header('Duplicate Question Pairs')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button('Find'):
    # Use helper's predict_duplicate function which handles preprocessing and feature creation
    result = helper.predict_duplicate(q1, q2)

    if result == 1:
        st.header('Duplicate means 1 , that means answer will be same ')
    else:
        st.header('Not Duplicate means 0, that means answer will be different')
