# dashboard.py
import streamlit as st
from agent import CodeSnippetAgent

# Initialize agent
agent = CodeSnippetAgent()

st.set_page_config(page_title="Code Snippet Generator", layout="wide")
st.title("🖥️ Code Snippet Generator Agent")
st.write("Generate Python code snippets and get explanations interactively.")

# User input
user_input = st.text_area("Enter your request (e.g., 'Python function to reverse a string'):")

if st.button("Generate Code"):
    if user_input.strip() != "":
        with st.spinner("Generating code..."):
            code = agent.generate_code(user_input)
            st.subheader("Generated Code:")
            st.code(code, language="python")

        # Explanation button
        if st.button("Explain Code"):
            with st.spinner("Generating explanation..."):
                explanation = agent.explain_code(code)
                st.subheader("Explanation:")
                st.write(explanation)
    else:
        st.warning("Please enter a request.")
