import requests
import streamlit as st
from typing import List

API_URL = "http://localhost:8000/generate/"

st.set_page_config(
    page_title="Intelligent Email Writer", 
    layout="centered"
)

st.title("📝 Intelligent Email Writer for Students")

# 1. kategori email
category = st.selectbox(
    "Kategori Email",
    [
        "Akademik",
        "Bimbingan & Skripsi",
        "Magang / MBKM",
        "Beasiswa / Exchange",
        "Organisasi / Kepanitiaan",
        "Karier & Profesional",
        "Umum & Administratif"
    ]
)

# 2. penerima
recipient = st.text_input(
    "Kepada",
    placeholder="e.g., Dosen Pembimbing, TU Fakultas, dst."
)

# 3. subjek
subject = st.text_input(
    "Subjek Email",
    placeholder="e.g., Permohonan Izin Tidak Hadir Kuliah"
)

# 4. tone penulisan
tone = st.selectbox(
    "Gaya/Tone Penulisan",
    ["Formal dan Sopan", "Santai namun Sopan", "Netral"]
)

# 5. bahasa
language = st.selectbox(
    "Bahasa",
    ["Bahasa Indonesia", "Bahasa Inggris"]
)

# 6. tingkat urgensi (opsional)
urgency = st.selectbox(
    "Tingkat Urgensi",
    ["Biasa", "Tinggi", "Rendah"]
)

# 7. poin-poin utama (pisah baris baru)
points_input = st.text_area(
    "Poin-poin Utama Isi Email",
    placeholder="Tuliskan poin-poin penting, satu poin per baris"
)
# ubah menjadi list
points = [p.strip() for p in points_input.split("\n") if p.strip()]

# 8. contoh email sebelumnya (opsional)
example = st.text_area(
    "Contoh Email Sebelumnya (Opsional)",
    height=100
)

# generate email
if st.button("✉️ Buat Email"):
    if not (recipient and subject and points):
        st.error("Mohon isi paling tidak: Kepada, Subjek, dan Poin-poin isi email.")
    else:
        # susun payload
        payload = {
            "category": category,
            "recipient": recipient,
            "subject": subject,
            "tone": tone,
            "language": language,
            "urgency_level": urgency,
            "points": points,
            "example_email": example
        }

        # kirim ke backend
        try:
            # kirim request ke API
            response = requests.post(API_URL, json=payload, timeout=15)
            
            # cek status code
            response.raise_for_status()
            
            # ambil data dari response
            data = response.json()
            
            # tampilkan hasil
            st.subheader("📄 Hasil Email")
            st.markdown(data.get("generated_email", "– Tidak ada output –"))
        except requests.exceptions.HTTPError as e:
            st.error(f"Server Error {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Gagal menghubungi server: {e}")