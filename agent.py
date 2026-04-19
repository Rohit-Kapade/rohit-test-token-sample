from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
PROMPTS_PATH = BASE_DIR / "prompts.json"


@dataclass
class StepResult:
    step_name: str
    prompt_rendered: str
    output_text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    token_source: str
    model_name: str


@dataclass
class RunResult:
    selected_model_key: str
    selected_model_label: str
    selected_prompt_key: str
    selected_prompt_name: str
    user_task: str
    steps: List[StepResult]

    @property
    def grand_total_tokens(self) -> int:
        return sum(step.total_tokens for step in self.steps)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["grand_total_tokens"] = self.grand_total_tokens
        return payload


class ConfigError(RuntimeError):
    pass


class MultiShotAgent:
    def __init__(
        self,
        config_path: Path = CONFIG_PATH,
        prompts_path: Path = PROMPTS_PATH,
        model_key: str | None = None,
        prompt_key: str | None = None,
    ):
        self.config = self._load_json(config_path)
        self.prompts = self._load_json(prompts_path)

        self.model_key = model_key or self.config["active_model_key"]
        self.prompt_key = prompt_key or self.config["active_prompt_key"]

        if self.model_key not in self.config["models"]:
            raise ConfigError(f"Unknown model key: {self.model_key}")
        if self.prompt_key not in self.prompts:
            raise ConfigError(f"Unknown prompt key: {self.prompt_key}")

        self.model_cfg = self.config["models"][self.model_key]
        self.prompt_cfg = self.prompts[self.prompt_key]
        self.llm = self._build_llm()

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _build_llm(self):
        provider = self.model_cfg["provider"]
        temperature = self.config.get("temperature", 0.2)
        max_completion_tokens = self.config.get("max_completion_tokens", 500)

        if provider == "openai":
            api_key_env = self.model_cfg.get("env_api_key", "OPENAI_API_KEY")
            if not os.getenv(api_key_env):
                raise ConfigError(
                    f"Missing environment variable {api_key_env} for OpenAI model access."
                )
            return ChatOpenAI(
                model=self.model_cfg["model_name"],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )

        if provider == "ollama":
            base_url = os.getenv(
                self.model_cfg.get("base_url_env", "OLLAMA_BASE_URL"),
                "http://localhost:11434",
            )
            return ChatOllama(
                model=self.model_cfg["model_name"],
                temperature=temperature,
                base_url=base_url,
            )

        raise ConfigError(f"Unsupported provider: {provider}")

    def _extract_token_usage(self, response_text: str, ai_msg: Any) -> Tuple[int, int, int, str]:
        usage = getattr(ai_msg, "usage_metadata", None) or {}
        if usage:
            input_tokens = int(usage.get("input_tokens", 0))
            output_tokens = int(usage.get("output_tokens", 0))
            total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens))
            return input_tokens, output_tokens, total_tokens, "provider"

        metadata = getattr(ai_msg, "response_metadata", None) or {}
        if "prompt_eval_count" in metadata or "eval_count" in metadata:
            input_tokens = int(metadata.get("prompt_eval_count", 0))
            output_tokens = int(metadata.get("eval_count", 0))
            total_tokens = input_tokens + output_tokens
            return input_tokens, output_tokens, total_tokens, "provider"

        input_tokens = (
            self.llm.get_num_tokens_from_messages(getattr(ai_msg, "_input_messages", []))
            if hasattr(self.llm, "get_num_tokens_from_messages")
            else 0
        )
        output_tokens = (
            self.llm.get_num_tokens(response_text)
            if hasattr(self.llm, "get_num_tokens")
            else max(1, len(response_text.split()))
        )
        total_tokens = input_tokens + output_tokens
        return input_tokens, output_tokens, total_tokens, "estimated"

    def _invoke_step(self, step_name: str, instructions: str, **variables: str) -> StepResult:
        template = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompt_cfg["system_prompt"]),
                ("human", instructions),
            ]
        )
        rendered_messages = template.format_messages(**variables)
        rendered_text = "\n\n".join(
            f"[{msg.type.upper()}]\n{msg.content}" for msg in rendered_messages
        )

        ai_msg = self.llm.invoke(rendered_messages)
        setattr(ai_msg, "_input_messages", rendered_messages)

        output_text = (
            ai_msg.content if isinstance(ai_msg.content, str) else json.dumps(ai_msg.content, indent=2)
        )
        input_tokens, output_tokens, total_tokens, token_source = self._extract_token_usage(output_text, ai_msg)

        model_name = (getattr(ai_msg, "response_metadata", {}) or {}).get(
            "model_name", self.model_cfg["model_name"]
        )

        return StepResult(
            step_name=step_name,
            prompt_rendered=rendered_text,
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            token_source=token_source,
            model_name=model_name,
        )

    def run(self, user_task: str) -> RunResult:
        step_1 = self._invoke_step(
            step_name="Step 1 - Plan",
            instructions=(
                "User goal: {user_task}\n\n"
                "Create a short plan with exactly 3 numbered bullets. Each bullet must contain:"
                " objective, why it matters, and what information should be passed to the next step."
            ),
            user_task=user_task,
        )

        step_2 = self._invoke_step(
            step_name="Step 2 - Execute",
            instructions=(
                "Original user goal: {user_task}\n\n"
                "Plan from previous step:\n{previous_output}\n\n"
                "Execute the plan and produce a detailed working draft with these headings:"
                " Assumptions, Draft Response, Risks."
            ),
            user_task=user_task,
            previous_output=step_1.output_text,
        )

        step_3 = self._invoke_step(
            step_name="Step 3 - Refine",
            instructions=(
                "Original user goal: {user_task}\n\n"
                "Working draft from previous step:\n{previous_output}\n\n"
                "Refine the draft into a polished final answer with these headings:"
                " Final Answer, Key Decisions, Next Best Action."
            ),
            user_task=user_task,
            previous_output=step_2.output_text,
        )

        return RunResult(
            selected_model_key=self.model_key,
            selected_model_label=self.model_cfg["label"],
            selected_prompt_key=self.prompt_key,
            selected_prompt_name=self.prompt_cfg["name"],
            user_task=user_task,
            steps=[step_1, step_2, step_3],
        )


def load_config_and_prompts() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)
    with PROMPTS_PATH.open("r", encoding="utf-8") as f:
        prompts = json.load(f)
    return config, prompts
