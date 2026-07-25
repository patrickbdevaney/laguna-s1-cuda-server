// chat.h — poolside_v1 chat grammar: rendering, reasoning separation, tool calls.
//
// Transcribed from models/…/chat_template.jinja (MODEL_INVENTORY.md §7). Replaces both
// `--reasoning-parser poolside_v1` and `--tool-call-parser poolside_v1`.
//
// Wire format:
//   〈|EOS|〉
//   <system>{sys}[\n\n### Tools\n\n…<available_tools>\n{one tool JSON per line}\n</available_tools>]</system>\n
//   <user>{content}</user>\n
//   <assistant><think>{reasoning}</think>{content}[<tool_call>…</tool_call>…]</assistant>\n
//   <tool_response>{content}</tool_response>\n
// generation prompt: "<assistant><think>" (thinking on) or "<assistant></think>" (off)
//
// Three things that are easy to get wrong and that this file exists to get right:
//
//  1. **Tool calls are NOT JSON on the wire.** They are
//     `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…</tool_call>`.
//     The template emits a raw string when the argument is a string and `tojson` otherwise,
//     so parsing has to recover the type. See `parse_arg_value`.
//
//  2. **Preserved thinking is the default.** With `enable_thinking` (shipped default true)
//     EVERY past assistant turn re-renders its `<think>…</think>`. If the server drops
//     reasoning from history, the rendered prefix diverges from what the model saw and the
//     KV prefix cache misses on every turn.
//
//  3. **An explicitly empty system message suppresses the default block**, which is
//     different from omitting the system message (that injects poolside's default text).
#pragma once
#include <string>
#include <vector>
#include <optional>
#include "third_party/json.hpp"

namespace lgchat {

// Tool JSON must be emitted EXACTLY as Jinja's `tojson` does, because it goes into the prompt
// verbatim and the model was trained on that byte sequence. Two differences from
// nlohmann's `dump()`: Python preserves object key INSERTION ORDER (nlohmann::json sorts
// alphabetically -> use ordered_json), and Python's default separators are ", " and ": "
// (nlohmann emits no spaces). Both were caught by the byte-exact gate.
using ojson = nlohmann::ordered_json;

inline void py_dump(const ojson& j, std::string& o) {
    if (j.is_object()) {
        o += '{';
        bool first = true;
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (!first) o += ", ";
            first = false;
            o += ojson(it.key()).dump();
            o += ": ";
            py_dump(it.value(), o);
        }
        o += '}';
    } else if (j.is_array()) {
        o += '[';
        for (size_t i = 0; i < j.size(); ++i) { if (i) o += ", "; py_dump(j[i], o); }
        o += ']';
    } else {
        o += j.dump();                      // scalars: nlohmann matches Python here
    }
}
inline std::string py_dump(const ojson& j) { std::string s; py_dump(j, s); return s; }

inline const char* kDefaultSystem =
    "You are a helpful, conversationally-fluent assistant made by Poolside. You are here to "
    "be helpful to users through natural language conversations.";

struct ToolCall {
    std::string id, name, arguments;      // `arguments` is a JSON object string (OpenAI shape)
};

struct Message {
    std::string role;                     // system | user | assistant | tool
    std::string content;
    std::string reasoning;                // assistant only
    std::vector<ToolCall> tool_calls;     // assistant only
};

inline std::string rstrip(const std::string& s) {
    size_t e = s.find_last_not_of(" \t\n\r\f\v");
    return e == std::string::npos ? std::string() : s.substr(0, e + 1);
}
inline bool strip_empty(const std::string& s) { return rstrip(s).empty(); }

// ---------------------------------------------------------------- rendering
// `tools` is an array of OpenAI tool objects (each already the JSON the model should see).
inline std::string render(const std::vector<Message>& msgs, const ojson& tools,
                          bool enable_thinking, bool add_generation_prompt,
                          bool preserve_thinking = false) {
    std::string out = "〈|EOS|〉";

    std::string system = kDefaultSystem;
    size_t first = 0;
    if (!msgs.empty() && msgs[0].role == "system") { system = msgs[0].content; first = 1; }

    const bool has_sys = !strip_empty(system);
    const bool has_tools = tools.is_array() && !tools.empty();
    if (has_sys || has_tools || enable_thinking) {
        out += "<system>";
        if (has_sys) {
            out += rstrip(system);
            if (has_tools) out += "\n\n";
        }
        if (has_tools) {
            out += "### Tools\n\n";
            out += "You may call functions to assist with the user query.\n";
            out += "All available function signatures are listed below:\n";
            out += "<available_tools>\n";
            for (const auto& t : tools) out += py_dump(t) + "\n";
            out += "</available_tools>";
        }
        out += "</system>\n";
    }

    for (size_t i = first; i < msgs.size(); ++i) {
        const Message& m = msgs[i];
        if (m.role == "user") {
            out += "<user>" + m.content + "</user>\n";
        } else if (m.role == "assistant") {
            out += "<assistant>";
            if (enable_thinking || preserve_thinking) out += "<think>" + m.reasoning + "</think>";
            else                                      out += "</think>";
            out += m.content;
            for (const auto& tc : m.tool_calls) {
                out += "<tool_call>" + tc.name;
                ojson args = ojson::object();
                try { args = ojson::parse(tc.arguments); } catch (...) {}
                if (args.is_object())
                    for (auto it = args.begin(); it != args.end(); ++it) {
                        out += "<arg_key>" + it.key() + "</arg_key><arg_value>";
                        // mirror the template: raw for strings, tojson otherwise
                        out += it.value().is_string() ? it.value().get<std::string>()
                                                      : it.value().dump();
                        out += "</arg_value>";
                    }
                out += "</tool_call>";
            }
            out += "</assistant>\n";
        } else if (m.role == "tool") {
            out += "<tool_response>" + m.content + "</tool_response>\n";
        } else if (m.role == "system") {
            out += "<system>" + m.content + "</system>\n";
        }
    }

    if (add_generation_prompt) {
        out += "<assistant>";
        out += enable_thinking ? "<think>" : "</think>";
    }
    return out;
}

// ---------------------------------------------------------------- parsing
// The template writes a bare string for string arguments and `tojson` for everything else,
// which is lossy: the literal text `3` could have been the number 3 or the string "3".
// Recover the most likely type — JSON scalars and containers parse, everything else is a
// string. Documented rather than hidden, because a tool expecting a string "3" and getting
// the number 3 is a real failure mode.
inline ojson parse_arg_value(const std::string& raw) {
    if (raw.empty()) return raw;
    const char c = raw[0];
    const bool looks_json = (c == '{' || c == '[' || c == '-' || (c >= '0' && c <= '9') ||
                             raw == "true" || raw == "false" || raw == "null");
    if (!looks_json) return raw;
    try {
        ojson v = ojson::parse(raw);
        if (v.is_number() || v.is_boolean() || v.is_null() || v.is_object() || v.is_array())
            return v;
    } catch (...) {}
    return raw;
}

struct Parsed {
    std::string reasoning;
    std::string content;
    std::vector<ToolCall> tool_calls;
};

// `text` is what the model generated AFTER the generation prompt. When thinking was on, the
// prompt already emitted "<think>", so the reasoning runs until the first "</think>".
inline Parsed parse_output(const std::string& text, bool thinking_was_on) {
    Parsed p;
    std::string body = text;

    if (thinking_was_on) {
        size_t e = body.find("</think>");
        if (e != std::string::npos) { p.reasoning = body.substr(0, e); body = body.substr(e + 8); }
        else { p.reasoning = body; body.clear(); }     // still inside the thought
    } else if (body.compare(0, 8, "</think>") == 0) {
        body = body.substr(8);
    }

    // strip a trailing </assistant> and anything after it
    size_t end = body.find("</assistant>");
    if (end != std::string::npos) body = body.substr(0, end);

    size_t pos = 0; int n = 0;
    while (true) {
        size_t s = body.find("<tool_call>", pos);
        if (s == std::string::npos) break;
        p.content += body.substr(pos, s - pos);
        size_t e = body.find("</tool_call>", s);
        std::string blk = body.substr(s + 11, (e == std::string::npos ? body.size() : e) - (s + 11));

        ToolCall tc;
        tc.id = "call_" + std::to_string(n++);
        size_t k = blk.find("<arg_key>");
        tc.name = blk.substr(0, k == std::string::npos ? blk.size() : k);
        ojson args = ojson::object();
        while (k != std::string::npos) {
            size_t ke = blk.find("</arg_key>", k);
            if (ke == std::string::npos) break;
            std::string akey = blk.substr(k + 9, ke - (k + 9));
            size_t vs = blk.find("<arg_value>", ke);
            if (vs == std::string::npos) break;
            size_t ve = blk.find("</arg_value>", vs);
            std::string aval = blk.substr(vs + 11, (ve == std::string::npos ? blk.size() : ve) - (vs + 11));
            args[akey] = parse_arg_value(aval);
            k = blk.find("<arg_key>", ve == std::string::npos ? vs : ve);
        }
        tc.arguments = py_dump(args);
        p.tool_calls.push_back(tc);
        if (e == std::string::npos) { pos = body.size(); break; }
        pos = e + 12;
    }
    p.content += body.substr(std::min(pos, body.size()));
    return p;
}

} // namespace lgchat
