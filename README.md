# LangChain Multi-Shot Agentic AI Demo

This example shows a simple **three-call sequential agentic workflow** using LangChain and Streamlit.

## What is included

- `agent.py` - the multi-shot workflow implementation
- `app.py` - Streamlit UI
- `config.json` - active model and runtime settings
- `prompts.json` - three prompt variants
- `requirements.txt` - Python dependencies

## Workflow

1. **Plan** - the first LLM call creates a concise task plan.
2. **Execute** - the second LLM call uses the plan output to produce a working draft.
3. **Refine** - the third LLM call uses the draft to produce a polished final response.

Each step captures token usage separately and the Streamlit app displays the result for every call.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### OpenAI models

Set your API key:

```bash
export OPENAI_API_KEY="your-key"
```

### Ollama model

Make sure Ollama is running locally and the model is pulled:

```bash
ollama pull llama3.2
export OLLAMA_BASE_URL="http://localhost:11434"
```

## Run the app

```bash
streamlit run app.py
```

## Change the model or prompt

Edit `config.json`:

- `active_model_key` can be `gpt-5-nano`, `gpt5-mini`, or `llama-3.2`
- `active_prompt_key` can be `analyst`, `coach`, or `executive`

## Notes on token usage

- For providers that return token usage metadata directly, the app displays the provider-reported token counts.
- For backends that do not expose usage metadata in the same format, the app falls back to a best-effort estimate and labels it as `estimated`.
