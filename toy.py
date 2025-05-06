#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = ["openai", "python-dotenv"]
# ///

import json
import os

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API base and model name from environment variables
openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
openai_api_model_name = os.getenv("OPENAI_API_MODEL_NAME", "gpt-4o")

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please add your OpenAI API key to the .env file.")

# Configure OpenAI API
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

def generate_problems(task_specification):
    """
    Generate programming problems based on a task specification.

    Args:
        task_specification (dict): A dictionary containing the task specification.

    Returns:
        dict: A JSON object containing the generated problems.
    """
    system_prompt = """You are a Parson’s problem generator designed for sentence construction in French.

Your task is to generate a set of sentence scramble problems focused on French grammar and syntax. Each problem must help learners practice assembling grammatically correct and semantically appropriate French sentences using given word or phrase blocks.

Each problem should include the following:

title: A short instructional phrase describing what kind of sentence the user is to construct. It must be in French.
solution: A list of words or phrase blocks in the correct order to form a valid French sentence.
distractors: A list of incorrect word or phrase blocks. These can include:
Incorrect verb conjugations (e.g., “vas” instead of “vais” for je)
Syntactic errors (e.g., inverted phrases like “école à”)
Wrong word choices (e.g., “tout” instead of “tous”)
Incorrect articles or prepositions (e.g., “à école” instead of “à l’école”)
Overspecified/underspecified fragments (e.g., “le déjeuner petit”)
Words from similar contexts but unrelated to the target sentence
Output format:
You should return a JSON object with the following structure:

{
"problems": [
{
"title": "...", // Brief French instruction
"solution": [ "...", ... ], // List of valid words/phrases in correct order
"distractors": [ "...", ... ] // List of plausible but incorrect words/phrases
},
...
]
}

Guidelines:

Focus on standard present-tense declarative sentences in French unless otherwise specified.
Choose everyday vocabulary suitable for language learners at the A1–B1 (beginner to intermediate) level.
Use a mix of subject pronouns (je, tu, il/elle, nous, vous, ils/elles), common verbs (aller, faire, avoir, être, prendre, etc.), and commonly used sentence structures.
The sentence should be communicative—something that could plausibly be said or written in a real-life context, such as daily routines, school, travel, weather, time, etc.
Use correct French punctuation and diacritics in all words (e.g., “école”, “huit”, “déjeuner”).
Each problem should be solvable by dragging and dropping the solution blocks into the correct order.
The generator must return exactly as many problems as requested in the prompt that follows this system message.

Do not include explanations in your output—only generate and return the JSON object as specified.
"""

    # Call OpenAI API
    chunks = client.chat.completions.create(
        model=openai_api_model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(task_specification)},
        ],
        response_format={"type": "json_object"},
        stream=True,
    )

    for chunk in chunks:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


if __name__ == "__main__":
    # Example task specification
    task_spec = {
        "language": "Python",
        "concepts": {
            "Easy": {"Variable Assignment": True, "Basic Arithmetic": True},
            "Medium": {"Functions": True},
            "Hard": {"Recursion": False},
        },
        "num_problems": 2,
    }

    # Generate problems
    deltas = generate_problems(task_spec)

    # Print the streaming output
    for delta in deltas:
        print(delta, end="")
