import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PdfReader(pdf_file)
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text


def preprocess_text(text):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    max_length = 512
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    elif len(tokens) < max_length:
        tokens += [0] * (max_length - len(tokens))
    tensor = torch.tensor([tokens])
    return tensor


def calculate_similarity(job_desc_embedding, resume_embedding):
    similarity_score = cosine_similarity(job_desc_embedding, resume_embedding)[0][0]
    return similarity_score


def main():
    st.title("Resume Matcher")

    job_description_pdf = st.file_uploader("Upload job description (PDF format)", type="pdf")
    resume_pdf = st.file_uploader("Upload a resume (PDF format)", type="pdf")

    if job_description_pdf and resume_pdf:
        job_description_text = extract_text_from_pdf(job_description_pdf)
        job_desc_tensor = preprocess_text(job_description_text)

        resume_text = extract_text_from_pdf(resume_pdf)
        resume_tensor = preprocess_text(resume_text)

        with torch.no_grad():
            job_desc_output = model(job_desc_tensor)
            resume_output = model(resume_tensor)

        job_desc_embedding = job_desc_output[0][:, 0, :].numpy()
        resume_embedding = resume_output[0][:, 0, :].numpy()

        similarity_score = calculate_similarity(job_desc_embedding, resume_embedding)
        similarity_percentage = round(similarity_score * 100, 2)

        st.subheader("Matching Resume:")
        st.write(f"Similarity Score: {similarity_percentage}%")
        st.write(f"Resume Path: {resume_pdf.name}")


if __name__ == "__main__":
    main()
