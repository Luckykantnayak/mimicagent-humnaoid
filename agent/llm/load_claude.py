import os

from langchain_anthropic import ChatAnthropic

from prompts.prompt_loader import load_yaml_config


def load_llm():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export your Anthropic API key first:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-..."
        )

    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg", "config.yaml")
    llm_cfg = load_yaml_config(cfg_path)["llm"]

    kwargs = dict(
        model_name=llm_cfg["model"],
        # Thinking tokens count against this ceiling, and the longest plans alone are
        # ~15k tokens. Streaming (below) keeps a large ceiling safe from HTTP timeouts.
        max_tokens_to_sample=int(llm_cfg.get("max_tokens", 64000)),
        api_key=api_key,
        # Long plans can exceed the read timeout before a non-streamed response
        # arrives. LangChain aggregates streamed chunks into the same AIMessage.
        streaming=True,
        # Ride out transient API errors (429/5xx/529) with the SDK's backoff.
        max_retries=8,
        timeout=float(llm_cfg.get("timeout_s", 600)),
        # No temperature/top_p/top_k: Claude Opus 5 rejects sampling parameters.
    )
    # Adaptive thinking: planning and coding are multi-step geometric reasoning. With
    # thinking disabled, Opus 5 can leak <thinking> text into the answer. BaseAgent
    # returns only the text blocks, so every node still receives a plain string.
    thinking = str(llm_cfg.get("thinking", "adaptive")).lower()
    if thinking in ("adaptive", "disabled"):
        kwargs["thinking"] = {"type": thinking}
    if llm_cfg.get("effort"):
        kwargs["effort"] = llm_cfg["effort"]
    # Server-side refusal fallback: if a safety classifier declines a request, the API
    # re-runs it on a fallback model inside the same call instead of stopping the run.
    if llm_cfg.get("refusal_fallback", True):
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["model_kwargs"] = {"fallbacks": "default"}

    return ChatAnthropic(**kwargs)


if __name__ == "__main__":
    llm = load_llm()
    resp = llm.invoke("Say 'OK' and nothing else.")
    print(resp.text)
    print(resp.response_metadata.get("stop_reason"), resp.usage_metadata)
