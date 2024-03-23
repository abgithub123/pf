
import streamlit as st
import openai

# Set your OpenAI API key
openai.api_key = "OPEN_API_KEY"

def analyze_document(content):
    """
    Analyzes the provided text content using OpenAI's API.
    """
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=content,
        max_tokens=150
    )

    return response.choices[0].text.strip()

def main():
    st.title("OpenAI Document Analysis")

    # File upload widget
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "docx", "pdf"])

    if uploaded_file is not None:
        file_contents = uploaded_file.read()

        st.subheader("Document Content:")
        st.write(file_contents.decode("utf-8"))

        # Button to trigger analysis
        if st.button("Analyze Document"):
            analysis_result = analyze_document(file_contents.decode("utf-8"))

            st.subheader("Analysis Results:")
            st.write(analysis_result)

if __name__ == "__main__":
    main()
