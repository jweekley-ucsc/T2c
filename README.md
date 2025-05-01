# Parsons Problem Generator API

This project is a FastAPI-based application for generating Parsons problems, which are educational programming exercises. The application uses OpenAI's GPT model to generate problems based on user-selected programming concepts and difficulty levels.

## Features

- **Dynamic Front-End**: A user-friendly front-end built with HTML and JavaScript, dynamically updates based on user input.
- **AI-Generated Problems**: Uses OpenAI's GPT model to generate Parsons problems tailored to the selected programming language, concepts, and difficulty.
- **GET API Endpoint**: A `/generate-problems` endpoint that accepts query parameters to generate problems.
- **Cross-Origin Support**: Configured with CORS middleware to allow requests from any origin.

## Project Structure

```
pyproject.toml       # Project configuration file
README.md            # Project documentation
server.py            # Main FastAPI application
uv.lock              # Dependency lock file
static/              # Static files (e.g., CSS, JS)
templates/           # HTML templates
    index.html       # Front-end interface
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies using `uv`:
   ```bash
   uv pip install -r pyproject.toml
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory with the following variables:
   ```env
   OPENAI_API_KEY=<your-openai-api-key>
   OPENAI_API_BASE=https://api.openai.com/v1
   OPENAI_API_MODEL_NAME=gpt-4
   ```

## Running the Application

1. Start the server in debug mode:
   ```bash
   uv run uvicorn server:app --reload
   ```

2. Open your browser and navigate to:
   - Front-end: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
   - API Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Usage

1. Use the front-end interface to select a programming language, concepts, and the number of problems.
2. The application dynamically generates a URL for the API request.
3. Click "Preview Result" to fetch and display the generated problems.
4. Copy the generated URL to use it in other tools or applications.

## API Reference

### `GET /generate-problems`

Generates Parsons problems based on the provided query parameter.

#### Query Parameter:
- `specification` (string): A base64-encoded JSON string containing the problem specification. The JSON object should have the following structure:
  ```json
  {
      "language": "Python",
      "concepts": {
          "Easy": {
              "Variable Assignment": true,
              "Basic Arithmetic": false
          },
          "Medium": {
              "Functions": true
          },
          "Hard": {
              "Recursion": false
          }
      },
      "num_problems": 3
  }
  ```

#### Example Request:
```bash
curl "http://127.0.0.1:8000/generate-problems?specification=eyJsYW5ndWFnZSI6ICJQeXRob24iLCAiY29uY2VwdHMiOiB7IkVhc3kiOiB7IlZhcmlhYmxlIEFzc2lnbm1lbnQiOiB0cnVlLCAiQmFzaWMgQXJpdGhtZXRpYyI6IGZhbHNlfSwgIk1lZGl1bSI6IHsiRnVuY3Rpb25zIjogdHJ1ZX0sICJIYXJkIjogeyJSZWN1cnNpb24iOiBmYWxzZX19LCAibnVtX3Byb2JsZW1zIjogM30="
```

The `specification` parameter is a base64-encoded string of the JSON object. You can use tools like `btoa` in JavaScript or `base64` in Python to encode the JSON object.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Written by [Adam Smith](https://adamsmith.as) using [GitHub Copilot agent mode](https://code.visualstudio.com/blogs/2025/02/24/introducing-copilot-agent-mode)
- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Powered by [OpenAI GPT](https://openai.com/)