import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables")

# Konfigurasi Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
MODEL = "gemini-2.0-flash"
model = genai.GenerativeModel(model_name=MODEL)

# Inisialisasi FastAPI
app = FastAPI(title="Intelligent Email Writer API")

# Schema request
class EmailRequest(BaseModel):
    category: str
    recipient: str
    subject: str
    tone: str
    language: str
    urgency_level: Optional[str] = "Biasa"
    points: List[str]
    example_email: Optional[str] = None

# Schema response
class EmailResponse(BaseModel):
    generated_email: str

# Fungsi builder prompt
def build_prompt(body: EmailRequest) -> str:
    lines = [
        f"Tolong buatkan email dalam {body.language.lower()} yang {body.tone.lower()}",
        f"kepada {body.recipient}.",
        f"Subjek: {body.subject}.",
        f"Kategori email: {body.category}.",
        f"Tingkat urgensi: {body.urgency_level}.",
        "",
        "Isi email harus mencakup poin-poin berikut:",
    ]
    for point in body.points:
        lines.append(f"- {point}")
    if body.example_email:
        lines += ["", "Contoh email sebelumnya:", body.example_email]
    lines.append("")
    lines.append("Buat email yang profesional, jelas, dan padat.")
    return "\n".join(lines)

# Endpoint utama
@app.post("/generate/", response_model=EmailResponse)
async def generate_email(req: EmailRequest):
    try:
        prompt = build_prompt(req)
        generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            top_k=32,
            max_output_tokens=2048,
        )

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
        )

        generated = response.text
        if not generated:
            raise HTTPException(status_code=500, detail="Tidak ada hasil dari Gemini API")

        return EmailResponse(generated_email=generated)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saat generate email: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "model": MODEL}
