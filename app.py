import streamlit as st
import nltk
import spacy
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from pathlib import Path

# Initialize NLP tools
nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Page config
st.set_page_config(page_title="🩸 Obsidian AI", layout="wide")
st.title("🩸 Obsidian AI — File Manager & Text Intelligence")
st.caption("By Nocturne — Designed in Shadow")

st.sidebar.header("Navigation")
section = st.sidebar.radio("Choose a Function:", ["Organize Files", "Summarize Text", "Generate Flashcards", "Analyze Text"])

if section == "Organize Files":
    st.header("📂 File Organizer")
    uploaded_files = st.file_uploader("Upload files to organize:", accept_multiple_files=True)
    if st.button("Organize Now"):
        output_dir = Path("OrganizedFiles")
        output_dir.mkdir(exist_ok=True)
        for uploaded_file in uploaded_files:
            with open(output_dir / uploaded_file.name, "wb") as f:
                f.write(uploaded_file.read())
        st.success(f"Files organized into: {output_dir.resolve()}")

elif section == "Summarize Text":
    st.header("📝 Summarize Text")
    text_input = st.text_area("Enter the text to summarize:")
    if st.button("Summarize"):
        if text_input:
            summary = summarizer(text_input, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            st.subheader("Summary:")
            st.success(summary)

elif section == "Generate Flashcards":
    st.header("📚 Generate Flashcards")
    flash_text = st.text_area("Enter educational text or notes:")
    if st.button("Generate Flashcards"):
        sents = sent_tokenize(flash_text)
        flashcards = []
        for sent in sents:
            doc = nlp(sent)
            keywords = [ent.text for ent in doc.ents]
            for kw in keywords:
                question = sent.replace(kw, "_____")
                flashcards.append(f"Q: {question}\nA: {kw}")
        if flashcards:
            st.subheader("Flashcards:")
            st.text("\n\n".join(flashcards))
        else:
            st.warning("No suitable flashcards found.")

elif section == "Analyze Text":
    st.header("🧠 Analyze Text")
    analyze_input = st.text_area("Enter text to analyze:")
    if st.button("Analyze"):
        doc = nlp(analyze_input)
        complexity = len(set([token.lemma_ for token in doc if not token.is_stop]))
        topics = ", ".join(set([ent.label_ for ent in doc.ents]))
        st.subheader("Analysis:")
        st.success(f"Topics: {topics or 'None found'}\nComplexity Score: {complexity}")

st.sidebar.markdown("---")
st.sidebar.caption("Built by Nocturne | Guided by Shade")
