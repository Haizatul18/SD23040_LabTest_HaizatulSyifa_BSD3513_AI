# text_chunking_app.py
import streamlit as st
import nltk
from PyPDF2 import PdfReader

# Download punkt tokenizer (for sentences)
nltk.download("punkt", quiet=True)

nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="PDF Text Chunker", layout="wide")
st.title("📄 PDF Text Chunking App")
st.caption(
    "Extract text from PDF, perform semantic chunking using NLTK sentence tokenizer, and view sentences."
)

# Upload PDF
uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded_file is not None:
    # Read PDF
    pdf_reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in pdf_reader.pages:
        raw_text += page.extract_text() + " "

    st.subheader("Extracted Text (Sample)")
    st.text_area("Full text", value=raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text, height=200)

    # ----------------------------
    # Sentence Tokenization
    # ----------------------------
    sentences = sent_tokenize(raw_text)

    st.subheader("Sentence Tokenization")
    st.write(f"Total sentences extracted: {len(sentences)}")

    # Show a sample of sentences (indices 58–68)
    start_idx = st.number_input("Start index for sample sentences", min_value=0, max_value=max(0, len(sentences)-1), value=58)
    end_idx = st.number_input("End index for sample sentences", min_value=start_idx, max_value=len(sentences), value=min(start_idx+10, len(sentences)))

    if st.button("Show Sample Sentences"):
        sample_sentences = sentences[start_idx:end_idx]
        st.write(sample_sentences)

    # ----------------------------
    # Optional: Word-based chunking
    # ----------------------------
    st.subheader("Optional: Word-based chunking")
    chunk_size = st.number_input("Number of words per chunk (N)", min_value=1, max_value=500, value=20, step=1)

    def word_chunker(text, N):
        words = text.split()
        chunks = []
        for i in range(0, len(words), N):
            chunks.append(" ".join(words[i:i+N]))
        return chunks

    if st.button("Create Word Chunks"):
        chunks = word_chunker(raw_text, chunk_size)
        st.success(f"Number of chunks created: {len(chunks)}")

        # Show first chunk
        st.subheader("Chunk 1")
        st.write(chunks[0])

        # Expandable view for all chunks
        with st.expander("Show All Chunks"):
            for i, ch in enumerate(chunks, start=1):
                st.markdown(f"**Chunk {i}**")
                st.write(ch)
                st.markdown("---")

else:
    st.info("Please upload a PDF file to start text chunking.")

