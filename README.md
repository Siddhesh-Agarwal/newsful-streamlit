# Newsful

This is a streamlit version of the newsful app.

## Execution

Add the following API keys to the `.streamlit/secrets.toml` file:

```toml
OPENAI_API_KEY=""
GEMINI_API_KEY=""
GOOGLE_CSE_ID=""
GOOGLE_API_KEY=""
```

Then install the dependencies:

```bash
pip install poetry && poetry install
```

Then run the app:

```bash
poetry run streamlit run main.py
```
