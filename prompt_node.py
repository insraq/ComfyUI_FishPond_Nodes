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
                "reasoning_budget": (
                    "INT",
                    {"default": 4096, "min": 0, "max": 100000, "step": 1},
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

    def generate(
        self, system_prompt, prompt, url, model="", api_key="", reasoning_budget=0
    ):
        endpoint = self._build_endpoint(url)

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "thinking_budget_tokens": reasoning_budget,
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

        # Print a labeled banner the first time each stream appears and whenever
        # the active stream switches, so thinking and output stay distinct. Each
        # stream gets its own color and body indentation for extra contrast.
        last_stream = [None]
        # ANSI styling: cyan for thinking, bold green for output. Banners use
        # ASCII only so they never trip UnicodeEncodeError on legacy consoles.
        styles = {
            "thinking": {"color": "\033[36m", "title": "[ THINKING ]"},
            "output": {"color": "\033[1;32m", "title": "[ OUTPUT ]"},
        }
        reset = "\033[0m"

        def emit(kind, text):
            if not text:
                return
            style = styles[kind]
            color = style["color"]
            if last_stream[0] != kind:
                if last_stream[0] is not None:
                    print(reset + "\n", flush=True)  # close prior stream + blank line
                bar = "=" * 60
                banner = ("\n{c}{bar}\n{c}{title}\n{c}{bar}{r}\n").format(
                    c=color, bar=bar, title=style["title"], r=reset
                )
                print(banner, end="", flush=True)
                last_stream[0] = kind
            # Re-apply the stream color for the body so it stays visually tied
            # to its banner even across many chunks.
            print(color + text + reset, end="", flush=True)

        with urllib.request.urlopen(req) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
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
                    emit("thinking", reasoning)

                piece = delta.get("content")
                if not piece:
                    continue

                # Split inline <think>...</think> reasoning from content.
                while piece:
                    if in_think_tag:
                        end = piece.find("</think>")
                        if end == -1:
                            thinking_parts.append(piece)
                            emit("thinking", piece)
                            piece = ""
                        else:
                            reason = piece[:end]
                            thinking_parts.append(reason)
                            emit("thinking", reason)
                            piece = piece[end + len("</think>") :]
                            in_think_tag = False
                    else:
                        start = piece.find("<think>")
                        if start == -1:
                            content_parts.append(piece)
                            emit("output", piece)
                            piece = ""
                        else:
                            content_parts.append(piece[:start])
                            emit("output", piece[:start])
                            piece = piece[start + len("<think>") :]
                            in_think_tag = True

        # Close the final stream with a separator so the block has a clear end.
        if last_stream[0] is not None:
            print(reset + "\n" + "=" * 60, flush=True)
        else:
            print()  # nothing streamed; just a trailing newline
        return ("".join(content_parts).strip(), "".join(thinking_parts).strip())
