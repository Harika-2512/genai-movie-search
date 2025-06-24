import streamlit as st
from embedding_service import embed_text
from search_service import search

st.title("🎬 GenAI Movie Search Engine")

query = st.text_input("Search for a movie...")

if query:
    with st.spinner("Searching..."):
        vector = embed_text(query)
        results = search(vector)

        st.success("Results:")
        for r in results:
            st.markdown(f"**{r['title']}**  \n{r['description']}  \n_Score: {r['score']:.2f}_")