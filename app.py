from __future__ import annotations

from pathlib import Path

import streamlit as st

from agent import ConfigError, MultiShotAgent, load_config_and_prompts

st.set_page_config(page_title="LangChain Multi-Shot Agent Demo", layout="wide")

BASE_DIR = Path(__file__).resolve().parent

st.title("LangChain Multi-Shot Agentic AI Demo")
st.caption("Three sequential LLM calls where each output becomes the next step's input.")

config, prompts = load_config_and_prompts()
model_options = config["models"]
prompt_options = prompts

def _safe_index(keys: list[str], desired: str) -> int:
    return keys.index(desired) if desired in keys else 0

model_keys = list(model_options.keys())
prompt_keys = list(prompt_options.keys())

with st.sidebar:
    st.subheader("Select options from JSON files")
    selected_model_key = st.selectbox(
        "Model from config.json",
        options=model_keys,
        index=_safe_index(model_keys, config.get("active_model_key", "")),
        format_func=lambda key: f"{model_options[key]['label']} ({key})",
    )
    selected_prompt_key = st.selectbox(
        "Prompt from prompts.json",
        options=prompt_keys,
        index=_safe_index(prompt_keys, config.get("active_prompt_key", "")),
        format_func=lambda key: f"{prompt_options[key]['name']} ({key})",
    )
    st.divider()
    st.write("These dropdowns are populated directly from `config.json` and `prompts.json`.")

selected_model = model_options[selected_model_key]
selected_prompt = prompt_options[selected_prompt_key]

st.subheader("Selections that will be used for the next run")
summary_cols = st.columns(2)
with summary_cols[0]:
    st.info(
        f"**Model selected**\n\n"
        f"- Key: `{selected_model_key}`\n"
        f"- Label: {selected_model['label']}\n"
        f"- Provider: {selected_model['provider']}\n"
        f"- Runtime model: `{selected_model['model_name']}`"
    )
with summary_cols[1]:
    st.info(
        f"**Prompt selected**\n\n"
        f"- Key: `{selected_prompt_key}`\n"
        f"- Name: {selected_prompt['name']}\n"
        f"- Description: {selected_prompt['description']}"
    )

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Prompt selected to run")
    st.json(selected_prompt, expanded=True)
with col2:
    st.subheader("LLM model selected to run")
    st.json(selected_model, expanded=True)

with st.expander("View full config.json", expanded=False):
    st.json(config, expanded=2)

with st.expander("View full prompts.json", expanded=False):
    st.json(prompts, expanded=2)

user_task = st.text_area(
    "Enter a task for the multi-shot agent",
    value="Create a concise launch plan for a new AI-powered meeting notes product aimed at small consulting firms.",
    height=110,
)

run_clicked = st.button("Run three-step agent", type="primary")

if run_clicked:
    try:
        agent = MultiShotAgent(model_key=selected_model_key, prompt_key=selected_prompt_key)
        result = agent.run(user_task.strip())
    except ConfigError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.exception(exc)
    else:
        st.success(
            f"Run complete using model '{result.selected_model_label}' and prompt '{result.selected_prompt_name}'."
        )

        metric_cols = st.columns(4)
        metric_cols[0].metric("LLM calls", len(result.steps))
        metric_cols[1].metric("Model", result.selected_model_label)
        metric_cols[2].metric("Prompt", result.selected_prompt_name)
        metric_cols[3].metric("Total tokens", result.grand_total_tokens)

        st.subheader("Sequential execution trace")
        for idx, step in enumerate(result.steps, start=1):
            with st.container(border=True):
                st.markdown(f"### {step.step_name}")
                info_cols = st.columns(5)
                info_cols[0].metric("Input tokens", step.input_tokens)
                info_cols[1].metric("Output tokens", step.output_tokens)
                info_cols[2].metric("Total tokens", step.total_tokens)
                info_cols[3].metric("Token source", step.token_source)
                info_cols[4].metric("Call #", idx)

                left, right = st.columns(2)
                with left:
                    st.markdown("**Prompt used for this call**")
                    st.code(step.prompt_rendered, language="text")
                with right:
                    st.markdown("**LLM output**")
                    st.write(step.output_text)

        st.subheader("Run result as JSON")
        st.json(result.to_dict(), expanded=3)

st.markdown("---")
st.markdown(
    "**How the chaining works:** Step 1 plans the task, Step 2 executes using Step 1 output, and Step 3 refines using Step 2 output."
)
