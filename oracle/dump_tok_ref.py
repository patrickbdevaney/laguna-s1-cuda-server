"""Reference token ids from the HF tokenizer, for gating the C++ ByteLevel BPE."""
import os, json, sys
from tokenizers import Tokenizer
MD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "models", "Laguna-S-2.1-NVFP4")
tk = Tokenizer.from_file(os.path.join(MD, "tokenizer.json"))
CASES = [
    "Hello, world!",
    "def fib(n):\n    return n if n < 2 else fib(n-1)+fib(n-2)\n",
    "  spaces   and\ttabs",
    "\n\n\nleading newlines",
    "trailing newlines\n\n\n",
    "numbers 1234567890 and 3.14159",
    "it's don't we're they've I'm you'll he'd",
    "IT'S DON'T WE'RE",                       # case-insensitive contraction branch
    "unicode: café naïve 日本語 emoji 🚀🔥 ",
    "punctuation!!! ??? ...--- ***",
    "〈|EOS|〉<system>You are helpful.</system>\n<user>hi</user>\n<assistant><think>",
    "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
    "mixed123abc456 CamelCaseWord snake_case_word",
    "a" * 200,
    "\t\t  \n  \t",
    "x  ",                                     # \\s+(?!\\S) trailing-space branch
    "Ω≈ç√∫˜µ≤≥÷",
    "```python\nprint('hi')\n```",
]
out = []
for s in CASES:
    ids = tk.encode(s, add_special_tokens=False).ids
    dec = tk.decode(ids, skip_special_tokens=False)
    out.append({"text": s, "ids": ids, "decoded": dec, "roundtrip": dec == s})
d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "kernel_refs")
os.makedirs(d, exist_ok=True)
json.dump(out, open(os.path.join(d, "tok_ref.json"), "w"), ensure_ascii=False, indent=1)
print(f"{len(out)} cases; roundtrip ok: {sum(c['roundtrip'] for c in out)}/{len(out)}")
for c in out[:4]: print(f"  {c['text']!r:40s} -> {c['ids'][:12]}")
