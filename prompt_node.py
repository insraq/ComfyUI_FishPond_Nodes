import json
import re
import urllib.request
from itertools import product


## ========Common Functions======== ##
def _split_top_level(text, sep="|"):
    """Split on `sep` only at the top brace level (ignores separators
    inside nested {...} groups)."""
    parts = []
    depth = 0
    current = []
    for ch in text:
        if ch == "{":
            depth += 1
            current.append(ch)
        elif ch == "}":
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


def _find_first_group(text):
    """Return (start, end) span of the first balanced {...} group, or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return (start, i)
    return None


def expand_wildcards(text):
    """Recursively expand {a|b} wildcard syntax into a list of all
    combinations. Supports nesting, e.g. {a|{b|c}}."""
    span = _find_first_group(text)
    if span is None:
        return [text]

    start, end = span
    prefix = text[:start]
    suffix = text[end + 1 :]
    inner = text[start + 1 : end]

    options = _split_top_level(inner, "|")

    results = []
    # Expand the suffix first so it can be combined with every option.
    for suffix_expanded in expand_wildcards(suffix):
        for option in options:
            for option_expanded in expand_wildcards(option):
                results.append(prefix + option_expanded + suffix_expanded)
    return results


## =========Node Classes========= ##
class PromptCombinations:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "This is a {blue|red} car in {New York|Miami}",
                    },
                ),
                "split_by_line": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("prompts", "count")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "combine"
    CATEGORY = "BillBum/Prompt"

    def combine(self, text, split_by_line=True):
        prompts = []
        if split_by_line:
            for line in text.splitlines():
                if line.strip() == "":
                    continue
                prompts.extend(expand_wildcards(line))
        else:
            prompts.extend(expand_wildcards(text))
        return (prompts, len(prompts))


class PromptGeneration:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are a helpful assistant that writes image generation prompts.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A cat sitting on a windowsill.",
                    },
                ),
                "url": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "http://localhost:8000/v1",
                    },
                ),
            },
            "optional": {
                "model": (
                    "STRING",
                    {"multiline": False, "default": ""},
                ),
                "api_key": (
                    "STRING",
                    {"multiline": False, "default": ""},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("generated_prompt", "thinking")
    FUNCTION = "generate"
    CATEGORY = "BillBum/Prompt"

    @staticmethod
    def _build_endpoint(url):
        """Normalize a base URL into a chat/completions endpoint."""
        url = url.strip().rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        return url + "/chat/completions"

    def generate(self, system_prompt, prompt, url, model="", api_key=""):
        endpoint = self._build_endpoint(url)

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
        }
        if model.strip():
            payload["model"] = model.strip()

        headers = {"Content-Type": "application/json"}
        if api_key.strip():
            headers["Authorization"] = "Bearer " + api_key.strip()

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        content_parts = []
        thinking_parts = []
        # Tracks whether we are inside an inline <think>...</think> block when
        # the server does not expose a dedicated reasoning field.
        in_think_tag = False

        with urllib.request.urlopen(req) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}

                # Dedicated reasoning field (DeepSeek / vLLM / others).
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    thinking_parts.append(reasoning)
                    print(reasoning, end="", flush=True)

                piece = delta.get("content")
                if not piece:
                    continue

                # Split inline <think>...</think> reasoning from content.
                while piece:
                    if in_think_tag:
                        end = piece.find("</think>")
                        if end == -1:
                            thinking_parts.append(piece)
                            print(piece, end="", flush=True)
                            piece = ""
                        else:
                            reason = piece[:end]
                            thinking_parts.append(reason)
                            print(reason, end="", flush=True)
                            piece = piece[end + len("</think>"):]
                            in_think_tag = False
                    else:
                        start = piece.find("<think>")
                        if start == -1:
                            content_parts.append(piece)
                            piece = ""
                        else:
                            content_parts.append(piece[:start])
                            piece = piece[start + len("<think>"):]
                            in_think_tag = True

        print()  # newline after streamed thinking
        return ("".join(content_parts).strip(), "".join(thinking_parts).strip())
