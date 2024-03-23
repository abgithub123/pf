import streamlit as st
from Summarization import SummarizationAssistant


def main():
    st.title("ANALYZE THIS")
    st.subheader("Open Lighthouse Plug In for analyzing files")

    # File upload widget
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "docx", "pdf"])

    #print(uploaded_file)
    # bytes_data = uploaded_file.read()
    # st.write(bytes_data)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask Me Anything (AMA)?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            assistant = SummarizationAssistant()
            bytes_data = uploaded_file.getvalue()
            #st.write(bytes_data)
            stream = assistant.ask(bytes_data)
            response = st.write(stream)
        st.session_state.messages.append({"role": "assistant", "content": stream})

if __name__ == "__main__":
    main()