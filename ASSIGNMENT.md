# T2c Assignment

In this assignment, create a tool that could be used by teachers and learners to generate structured content that is compatible with the Parsons problem format you developed for the T2b assignment. The biggest difference between T2b and T2c is that there is an AI component to your deployed system.

## Learning Objectives

- Working with GenAI
    - Use GenAI via API rather than UI
    - Generate prompts rather than hand-typing them
- Teaching with GenAI
    - Create content for a pre-existing platform (your T2b app)
    - Give others an traditional UI to control a GenAI-based generator
- Deploying GenAI-based systems
    - Run server-side code to power a client-side interactive experience
    - Hide your API key!

## Expected Outcome

By the end of the T2c assignment, you will have developed an interactive web application comparible this this example: [https://glitch.com/~parsons-problem-generator](https://glitch.com/~parsons-problem-generator). If offers a simple user interface in the browser for defining a problem generation task, and it makes uses of a simple web server to carry out the task using a LLM inference service. The server-side code is written in Python and uses the [FastAPI](https://fastapi.tiangolo.com/) framework to handle HTTP requests. The client-side code is written in JavaScript and uses the [Fetch](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API) API to make requests to the server.

Your project will likely be created by *remixing* the example project.

## Steps


### Gather your access credentials for an LLM provider

Easy way (using BayLeaf):
- In the root directory of this repository, create a file called `.env` and paste the following lines into it:
```
OPENAI_API_BASE=https://bayleaf.chat/openai
OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
OPENAI_API_MODEL_NAME=openai.gpt-4o
```
- Log into [BayLeaf Chat](https://bayleaf.chat/), find the Settings panel, visit the Account tab, and use the "API Key" option to generate and reveal an API key. It will be a long string that starts with `sk-`. Use your key to replace the placeholder text above.
- Try your best not to share this API key with anyone else.

Hard way (using a commercial LLM provider):
- Sign up for an account with a commercial LLM provider (e.g. OpenAI, Cohere, Anthropic, etc.). You may need to provide a credit card and may be charged for usage.
- Create a `.env` file as in the BayLeaf example above, but use the API key and model name for your provider. For example, if you are using OpenAI, you would use `gpt-4o` instead of `openai.gpt-4o`.

### Prepare your local development environment

1. On GitHub, use the starter code as a template for your own new repository. Look for the green "Use this template" button in the top right corner of the page: https://github.com/rndmcnlly/T2c
2. Make sure you have [VS Code](https://code.visualstudio.com/) installed or know how to access a cloud version (e.g. GitHub CodeSpaces).
3. Clone this repository and open it in VS Code: https://github.com/rndmcnlly/T2c
4. If you are prompted to install additional extensions (e.g. Python, GitHub Copilot, and Code Tour), do so.
5. Open a terminal in VS Code and ensure you can run the `uv` command. This command is part of a Python package manager used by the T2c starter code. If you don't have it installed, follow the instructions here: https://github.com/astral-sh/uv
6. In the terminal, run `uv run toy.py`. The first time you run this command, you will see a lot of output as the software dependencies are installed. If the run is successful, you should see a streaming description of the generated Parsons problem collection.

### Adapt the toy generator to match your T2b app

Revisit the JSON problem set specification you developed for the T2b assignment. Incrementally modify the simplified problem generator in `toy.py` to produce content compatible with your T2b app. You should be able to copy and paste the output from `toy.py` into your T2b app on Glitch and interact with freshly generated Parsons problems.

The first thing to change in `toy.py` to change should be the format description included in the text of `system_prompt`. This is how the LLM knows that needs to go into one of your Parsons problem sequences.

Next, you should revise the `task_spec` object to be relevant to your application. As you add or remove details from the generation task specification, think about what kind of graphical user interface would be appropriate for configuring the generator. Will there be checkboxes, drop-down menus, sliders, etc.?

At the end of this step, your version of `toy.py` that a hypothetical teacher or learner could use to generate additional content for your T2b app.

Now would be a good time to commit your changes to `toy.py` and push them back to your GitHub repository. (This is not a requirement of the assignment, but it is a good idea.)

### Build a (local) web service version of your generator

Revisit the design of `server.py` and `templates/index.html` that we saw during lecture. Optionally, use the *Code Tour* feature of VS Code to review step through the guided tours. The "Walkthrough" tour will show you how a generation task gets processed, and the "Customization" tour will show where you will need to modifiy the starter code.

Run a local version of the web service with this command:
```bash
uv run uvicorn server:app --reload
```

We are looking for lines of output that look like this:
```
INFO:    Uvicorn running on http://127.0.0.1:8000 
...
INFO:    Application startup complete.
```

Load http://127.0.0.1:8000 in your web browser to find the web interface to your generator. When you press the "Preview Result" button in the UI, expect that it takes a few seconds to generate the result. The result will only display once generated has completed (unlike the streaming outputs you saw in the terminal when running `toy.py`).

Now, try making a simple change to `index.html`. Perhaps you could change the name "Parsons" to "Farsons" or something equally silly. Save the file, and reload the page in your web browser. You should see your change reflected in the web page.

Next, use your modifications to `toy.py` to adapt the value of `system_prompt` in `server.py`. If the outputs look good, proceed to make changes to `index.html` to offer a user interface suitable for defining your version of the generation task specification.

If your generator seems to be working well, now would be a good time to commit your changes to `server.py` and `index.html` and push them back to your GitHub repository. (Again, this is not a requirement of the assignment, but it is a good idea.)

### Deploy your web service to Glitch

1. Remix the example project on Glitch: https://glitch.com/~parsons-problem-generator
2. In your remix, edit the `.env` file with the details you have been using in your local development environment.
3. Replace the contents of `server.py` and `index.html` with your own versions.
4. If the web page preview in Glitch seems to be stuck in the "Starting up..." state, try clicking the "LOGS" button in the bottom left corner of the Glitch editor. This will show you the server logs, which may help you diagnose the problem. In rare cases, you may need to press the "TERMINAL" button in the same menu bar to access a command line interface. You can use the command `killall uvicorn` to kill the misconfigured web server. A few seconds later, you should see messages in the Glitch logs that show similar to those you saw in VS Code.
5. Return to the assignment page on Canvas and share the URL of your Glitch remix with the course community. The assignment page will detail any other items that need to be submitted.