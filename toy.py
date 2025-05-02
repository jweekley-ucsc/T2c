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
    system_prompt = """You are a Parsons problem generator.

Your task is to generate a set of problems based on the selected concepts and programming language.
Each problem should include a problem statement, a solution, and distractor blocks.
It is okay if the distractor blocks are not complete or contain duplicates of the solution blocks. The problem display interface will handle that. In particular, the distractor set will be formed by the union of lines in the distractor blocks differenced by the set of solution blocks.
Use "_thoughts" to sketch out the problem before writing the detailed specification for it.

The output should be a JSON object with the following structure:

{
    "language": "JavaScript",
    "problems": [
        {
            "_thoughts": [
                "I'll require the user to build a function that doesn't get called.",
                "I'll test their ability to give it a good name.",
                "And I'll have them pick a good body for it.",
                "Distractors will include other names and bodies as well as loose body fragments with bad indentation."
            ],
            "problem": "Write a function that adds two numbers.",
            "solution_blocks": [
                "function add(a, b) {",
                "    return a + b;",
                "}"
            ],
            "distractor_blocks": [
                "function subtract(a, b) {",
                "    return a - b;",
                "}",
                "function multiply(a, b) {",
                "    return a * b;",
                "}",
                "a + b",
                "return a;"
            ],
            "difficulty": "Easy",
            "concepts": ["Variable Assignment", "Basic Arithmetic"]
        },
        ...
    ]
}

The problems should be relevant to the selected concepts without including any of the concepts that were not selected.
The collection should have exactly as many problems as specified in the JSON object that will follow.
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
