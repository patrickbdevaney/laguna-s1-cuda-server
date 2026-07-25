"""Reference renderings from the SHIPPED chat_template.jinja, to gate include/chat.h."""
import os, json
from jinja2 import Environment, BaseLoader
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "models", "Laguna-S-2.1-NVFP4")
src = open(os.path.join(MD, "chat_template.jinja")).read()
env = Environment(loader=BaseLoader(), trim_blocks=False, lstrip_blocks=False,
                  extensions=["jinja2.ext.do"])
# transformers ships its own `tojson` that takes ensure_ascii; stock jinja2's does not.
def _tojson(value, indent=None, ensure_ascii=False, **kw):
    return json.dumps(value, indent=indent, ensure_ascii=ensure_ascii,
                      separators=(", ", ": ") if indent is None else None)
env.filters["tojson"] = _tojson
# the template uses {% generation %} (a transformers extension); neutralise it for rendering
src = src.replace("{%- generation -%}", "").replace("{%- endgeneration -%}", "")
tpl = env.from_string(src)

TOOLS = [{"type": "function", "function": {"name": "get_weather",
          "description": "Get weather", "parameters": {"type": "object",
          "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

CASES = [
 ("plain", {"messages":[{"role":"user","content":"hi"}], "add_generation_prompt":True,
            "enable_thinking":True}),
 ("no_think", {"messages":[{"role":"user","content":"hi"}], "add_generation_prompt":True,
               "enable_thinking":False}),
 ("custom_sys", {"messages":[{"role":"system","content":"You are terse."},
                             {"role":"user","content":"hi"}],
                 "add_generation_prompt":True, "enable_thinking":True}),
 ("empty_sys", {"messages":[{"role":"system","content":""},
                            {"role":"user","content":"hi"}],
                "add_generation_prompt":True, "enable_thinking":False}),
 ("multiturn", {"messages":[{"role":"user","content":"hi"},
                            {"role":"assistant","content":"Hello!","reasoning":"They greeted me."},
                            {"role":"user","content":"bye"}],
                "add_generation_prompt":True, "enable_thinking":True}),
 ("tools", {"messages":[{"role":"user","content":"weather in Paris?"}], "tools":TOOLS,
            "add_generation_prompt":True, "enable_thinking":True}),
 ("toolcall_roundtrip", {"messages":[
        {"role":"user","content":"weather in Paris?"},
        {"role":"assistant","content":"","reasoning":"Need the tool.",
         "tool_calls":[{"type":"function","function":{"name":"get_weather",
                        "arguments":{"city":"Paris","days":3}}}]},
        {"role":"tool","content":"{\"temp\": 18}"}],
      "tools":TOOLS, "add_generation_prompt":True, "enable_thinking":True}),
]
out=[]
for name, kw in CASES:
    txt = tpl.render(**kw)
    out.append({"name":name, "kw":kw, "rendered":txt})
json.dump(out, open(os.path.join(ROOT,"docs","kernel_refs","chat_ref.json"),"w"),
          ensure_ascii=False, indent=1)
print(f"{len(out)} chat cases rendered")
print(repr(out[0]["rendered"][:120]))
print(repr(out[6]["rendered"][-220:]))
