import os
import json
import base64
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException

from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API configuration
openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
openai_api_model_name = os.getenv("OPENAI_API_MODEL_NAME", "gpt-4o")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("Please add your OpenAI API key to the .env file.")

# Configure OpenAI client
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development purposes)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static and template directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/generate-problems")
def generate_problems(specification: str):
    try:
        # Decode the base64-encoded user specification
        decoded_spec = base64.b64decode(specification).decode("utf-8")
        user_spec = json.loads(decoded_spec)

        # Get user preferences
        target_language = user_spec.get("language", "French").strip().capitalize()
        num_problems = user_spec.get("num_problems", 3)

        # Define vocab for different Romance languages
        language_vocab = {
            "French": {
                "examples": "passé composé, le déjeuner, l’école, il fait beau",
                "level_reference": "A1–B1",
                "article_example": "à l’école"
            },
            "Spanish": {
                "examples": "pretérito perfecto, desayuno, la escuela, hace buen tiempo",
                "level_reference": "A1–B1",
                "article_example": "a la escuela"
            },
            "Italian": {
                "examples": "passato prossimo, colazione, la scuola, fa bel tempo",
                "level_reference": "A1–B1",
                "article_example": "a scuola"
            }
        }

        if target_language not in language_vocab:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {target_language}")

        vocab = language_vocab[target_language]

        # Construct the system prompt for the LLM
        system_prompt = f"""
You are a Parson’s problem generator designed for sentence construction in {target_language}.
Your task is to generate a set of sentence scramble problems focused on grammatical and syntactic knowledge in {target_language}. Each problem must help learners practice assembling grammatically correct and semantically appropriate {target_language} sentences using given word or phrase blocks.

Each problem must include the following fields:
- title: A short instructional phrase in {target_language}
- solution: A list of words/phrases in correct order to form a valid declarative sentence
- distractors: Plausible but incorrect alternatives, such as:
  - Verb conjugation errors
  - Incorrect word order
  - Errors in gender/number agreement
  - Incorrect prepositions (e.g., "{vocab['article_example']}")

Output only a JSON object with the following structure:
{{
  "problems": [
    {{
      "title": "...",
      "solution": ["...", "..."],
      "distractors": ["...", "..."]
    }},
    ...
  ]
}}

Guidelines:
- Generate exactly {num_problems} problems
- Use standard present-tense declarative sentences (unless conceptually needed otherwise)
- Keep sentence topics suitable for {vocab['level_reference']} learners
- Include correct diacritics in {target_language} (e.g., {vocab['examples']})
- Do not add any instructional text or explanations — only the JSON object
"""

        # Optional user prompt (can include difficulty/concepts later)
        user_prompt = f"Please generate {num_problems} scrambled sentence problems in {target_language}."

        # Request from OpenAI
        response = client.chat.completions.create(
            model=openai_api_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )

        ai_response = response.choices[0].message.content
        problems = json.loads(ai_response)
        return JSONResponse(content=problems)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating problems: {str(e)}")
