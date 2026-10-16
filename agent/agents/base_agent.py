# agents/base_agent.py
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Appended to the request when the previous answer hit the output-length ceiling.
# A truncated answer is never passed on: a cut-off skill file can still parse and
# then silently run with inherited default callbacks.
TRUNCATION_NOTE = ("\n\nYour previous answer was cut off at the output-length limit before it "
                   "finished. Answer again from the beginning, completely, and more concisely.")


USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def merge_usage(total, *responses):
    """Accumulate responses' token usage (including prompt-cache reads and writes)
    into a run total, so metadata.json shows what caching actually saved."""
    out = {k: int((total or {}).get(k, 0) or 0) for k in USAGE_KEYS}
    for response in responses:
        usage = (getattr(response, "response_metadata", None) or {}).get("usage") or {}
        for k in USAGE_KEYS:
            out[k] += int(usage.get(k, 0) or 0)
    return out


def _response_text(response):
    """Text blocks only. Thinking blocks are dropped; every node parses plain text."""
    content = response.content
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content
                   if isinstance(block, dict) and block.get("type") == "text")


class BaseAgent:
    def __init__(self, llm, max_truncation_retries=2):
        self.llm = llm
        self.max_truncation_retries = max_truncation_retries

    @staticmethod
    def _prepare(messages):
        """Accept a plain prompt string or a message list. A system prompt is marked
        for prompt caching: it is identical across every call of the same node, so
        repair iterations and parallel runs re-read it from cache."""
        if isinstance(messages, str):
            return [HumanMessage(content=messages)]
        prepared = []
        for message in messages:
            if isinstance(message, SystemMessage) and isinstance(message.content, str) and message.content:
                message = SystemMessage(content=[{
                    "type": "text", "text": message.content,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}}])
            prepared.append(message)
        return prepared

    @staticmethod
    def _with_note(messages, note):
        messages = list(messages)
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage) and isinstance(messages[i].content, str):
                messages[i] = HumanMessage(content=messages[i].content + note)
                return messages
        return messages + [HumanMessage(content=note.strip())]

    def generate(self, messages) -> AIMessage:
        prepared = self._prepare(messages)
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                 "cache_creation_input_tokens": 0}
        response, stop = None, None
        for attempt in range(self.max_truncation_retries + 1):
            response = self.llm.invoke(prepared)
            meta = response.usage_metadata or {}
            usage["input_tokens"] += int(meta.get("input_tokens", 0) or 0)
            usage["output_tokens"] += int(meta.get("output_tokens", 0) or 0)
            details = meta.get("input_token_details") or {}
            usage["cache_read_input_tokens"] += int(details.get("cache_read", 0) or 0)
            usage["cache_creation_input_tokens"] += int(details.get("cache_creation", 0) or 0)
            stop = response.response_metadata.get("stop_reason")
            if stop != "max_tokens":
                break
            print(f"[llm] answer truncated at the output limit "
                  f"(attempt {attempt + 1}/{self.max_truncation_retries + 1})", flush=True)
            prepared = self._with_note(prepared, TRUNCATION_NOTE)
        metadata = dict(response.response_metadata)
        metadata["usage"] = usage
        metadata["truncated"] = stop == "max_tokens"
        metadata["refused"] = stop == "refusal"
        return AIMessage(content=_response_text(response), response_metadata=metadata,
                         usage_metadata=response.usage_metadata, id=response.id)
