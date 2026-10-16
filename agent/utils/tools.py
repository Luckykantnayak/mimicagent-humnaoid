import json
import re


def _strip_response_prefix(text: str) -> str:
    text = re.sub(r"^\s*AI:\s*", "", text)
    # Some providers return a completed reasoning block in the text channel even
    # with thinking disabled. Parse only the answer after that block, never JSON
    # examples inside it. An unclosed block remains invalid rather than salvaged.
    return re.sub(r"\A\s*(?:(?:<thinking>.*?</thinking>|<think>.*?</think>)\s*)+",
                  "", text, flags=re.DOTALL)


def extract_json_from_message(text: str):
    """
    Extract JSON from an LLM message that may contain:
    - 'AI:' prefix
    - ```json ... ``` fences
    """
    # Remove leading "AI:" if present
    text = _strip_response_prefix(text)

    # Extract fenced JSON if present
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    # Strip whitespace
    text = text.strip()

    # Validate / parse JSON. strict=False tolerates raw control characters (a literal
    # newline/tab inside a string value) that LLMs occasionally emit — otherwise a
    # single stray control char in a long description string crashes the whole skill
    # (a JSONDecodeError "Invalid control character at ..."). The data is unaffected.
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        # TRAILING COMMAS are the other routine LLM JSON quirk, and this parser is on
        # EVERY node's path (formatter, planner, diagnosis), so one stray comma aborts
        # the whole skill before any work happens — measured: a cartwheel run died 20 s
        # in at `prepare_skill` with "Illegal trailing comma before end of object",
        # having produced nothing. Strip commas that sit before a closing brace/bracket
        # (never inside a string) and retry once; if it still fails, raise the ORIGINAL
        # error so a genuinely malformed response is still reported honestly.
        repaired, out, in_str, esc = [], [], False, False
        for ch in text:
            if in_str:
                out.append(ch)
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                out.append(ch)
                continue
            if ch in "}]":
                while out and out[-1].isspace():
                    out.pop()
                if out and out[-1] == ",":
                    out.pop()
            out.append(ch)
        repaired = "".join(out)
        return json.loads(repaired, strict=False)


def extract_python_code_with_report(text: str):
    """(code, dropped): `dropped` is whatever had to be cut to make the file parse.
    Trailing prose is harmless; trailing CODE means the file was truncated or broken
    and must not be executed with the missing callbacks silently inherited."""
    text = _strip_response_prefix(text)
    fence_match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    text = text.strip()
    code = _largest_parseable_prefix(text)
    return code, text[len(code):].strip() if text.startswith(code) else ""


def dropped_code_lines(dropped: str):
    """Lines of `dropped` that look like code rather than prose."""
    return [l for l in dropped.splitlines()
            if re.match(r"^\s+\S", l) or re.match(r"^(def|class|return|if|elif|else|for|while|import|from|"
                                                   r"try|except|with|self\.|[A-Za-z_][A-Za-z0-9_]*\s*=)", l.strip())]


def extract_python_code_from_message(text: str):
    """
    Extract python code from an LLM message that may contain:
    - 'AI:' prefix
    - ```python ... ``` fences
    """
    # Remove leading "AI:" if present
    text = _strip_response_prefix(text)

    # Extract fenced python if present
    fence_match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    # Strip whitespace
    text = text.strip()

    # SALVAGE a syntactically-valid file. LLMs sometimes leak reasoning prose
    # (even in another language) after the code, INSIDE the fence — which makes the
    # whole file a SyntaxError and wastes an entire repair iteration on a non-bug.
    # If the text doesn't parse, keep the LARGEST leading prefix that does (drops
    # trailing non-code); if even that fails, return as-is so the runtime error is
    # still surfaced honestly.
    return _largest_parseable_prefix(text)


def _largest_parseable_prefix(text: str) -> str:
    import ast
    try:
        ast.parse(text)
        return text
    except SyntaxError:
        pass
    lines = text.split("\n")
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end])
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    return text
