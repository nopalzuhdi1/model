import os
import streamlit as st
from transformers import pipeline, TapasForQuestionAnswering, TapasTokenizer
import docx
import pytesseract
from PIL import Image
import numpy as np
import pandas as pd
import torch
from symspellpy import SymSpell, Verbosity

# Setel path ke Tesseract (untuk Streamlit Cloud)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Inisialisasi model summarization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summarizer = pipeline('summarization', model='facebook/bart-large-cnn', device=0 if device.type == 'cuda' else -1)

# Inisialisasi SymSpell untuk koreksi teks otomatis
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "/usr/share/dict/words"  # Path ke kamus (sesuaikan dengan sistem Anda)
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Inisialisasi model TAPAS untuk table question answering
tapas_model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq").to(device)
tapas_tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")

# Fungsi untuk membaca file teks
def read_txt(file):
    with open(file.name, 'r', encoding='utf-8') as f:
        return f.read()

# Fungsi untuk membaca file docx
def read_docx(file):
    doc = docx.Document(file.name)
    full_text = [para.text for para in doc.paragraphs]
    return '\n'.join(full_text)

# Fungsi untuk membersihkan dan mengoreksi teks dari OCR
def clean_and_correct_ocr_text(text):
    # Koreksi otomatis menggunakan SymSpell
    corrected_text = []
    for word in text.split():
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected_text.append(suggestions[0].term)
        else:
            corrected_text.append(word)
    return ' '.join(corrected_text)

# Fungsi untuk meringkas dokumen
def summarize_file(file):
    if file.name.endswith('.txt'):
        text = read_txt(file)
    elif file.name.endswith('.docx'):
        text = read_docx(file)
    else:
        return "Unsupported file type. Please upload a .txt or .docx file."

    # Bersihkan dan koreksi teks jika diperlukan
    text = clean_and_correct_ocr_text(text)

    # Bagi teks menjadi chunk yang lebih kecil
    chunks = [text[i:i + 1024] for i in range(0, len(text), 1024)]
    summarized_chunks = [summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return ' '.join(summarized_chunks)

# Fungsi untuk mengubah gambar menjadi teks menggunakan OCR
def image_to_text(image):
    try:
        # Menggunakan pytesseract untuk melakukan OCR pada gambar
        text = pytesseract.image_to_string(image)
        # Bersihkan dan koreksi teks dari OCR
        text = clean_and_correct_ocr_text(text)
        return text
    except Exception as e:
        return f"Error in image to text: {str(e)}"

# Fungsi untuk menjawab pertanyaan berdasarkan tabel
def table_question_answering(file, question):
    try:
        # Baca file CSV
        df = pd.read_csv(file.name)

        # Pastikan pertanyaan adalah string yang valid
        if not isinstance(question, str) or not question.strip():
            return "Please enter a valid question."

        # Pastikan tabel tidak kosong
        if df.empty:
            return "The table is empty. Please upload a valid CSV file."

        # Tokenisasi tabel dan pertanyaan
        inputs = tapas_tokenizer(table=df, queries=[question], padding="max_length", return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Lakukan prediksi
        outputs = tapas_model(**inputs)
        predicted_answer = tapas_tokenizer.convert_logits_to_answer(df, inputs, outputs.logits[0].detach().cpu().numpy())

        return predicted_answer
    except Exception as e:
        return f"Error in table question answering: {str(e)}"

# Fungsi untuk memproses input berdasarkan pilihan
def process_input(choice, file=None, image=None, ts_file=None, question=None):
    try:
        if choice == "Summarize Document":
            if file is not None:
                return summarize_file(file)
            else:
                return "Please upload a .txt or .docx file."
        elif choice == "Image to Text":
            if image is not None:
                return image_to_text(image)
            else:
                return "Please upload an image."
        elif choice == "Table Question Answering":
            if ts_file is not None and question is not None:
                return table_question_answering(ts_file, question)
            else:
                return "Please upload a CSV file and enter a question."
    except Exception as e:
        return f"Error in processing input: {str(e)}"

# Antarmuka Streamlit
st.title("MODEL NLP UNIMODEL BY NOPALZ")

choice = st.radio(
    "Choose Input Type",
    options=["Summarize Document", "Image to Text", "Table Question Answering"]
)

if choice == "Summarize Document":
    file = st.file_uploader("Upload Document (.txt or .docx)", type=["txt", "docx"])
    if file is not None:
        output = process_input(choice, file=file)
        st.write("Summary:")
        st.write(output)

elif choice == "Image to Text":
    image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if image is not None:
        image = Image.open(image)
        output = process_input(choice, image=image)
        st.write("Extracted Text:")
        st.write(output)

elif choice == "Table Question Answering":
    ts_file = st.file_uploader("Upload Table CSV", type=["csv"])
    question = st.text_input("Enter your question")
    if ts_file is not None and question:
        output = process_input(choice, ts_file=ts_file, question=question)
        st.write("Answer:")
        st.write(output)