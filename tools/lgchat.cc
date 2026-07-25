// lgchat — terminal client for the Laguna server. C++ only; no Python, no node, no curl.
//
// Streams the SSE response and prints reasoning dim and content bright, so a long chain of
// thought is visually separable from the answer without scrolling back. Keeps the whole
// conversation client-side and posts it every turn, which is exactly what the server's prefix
// cache is designed for: only the new turn gets prefilled.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include "../include/third_party/httplib.h"
#include "../include/third_party/json.hpp"

using json = nlohmann::ordered_json;

static const char* DIM = "\033[2m";
static const char* PURPLE = "\033[35m";
static const char* RESET = "\033[0m";
static const char* BOLD = "\033[1m";

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = 8080;
    double temp = 0.0;
    int max_tokens = 1024;
    bool thinking = true, color = true, show_think = true;
    std::string system_prompt, one_shot;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--host") host = next();
        else if (a == "--port") port = atoi(next());
        else if (a == "--temp") temp = atof(next());
        else if (a == "--max") max_tokens = atoi(next());
        else if (a == "--system") system_prompt = next();
        else if (a == "--no-think") thinking = false;
        else if (a == "--hide-think") show_think = false;
        else if (a == "--no-color") color = false;
        else if (a == "-p" || a == "--prompt") one_shot = next();
        else if (a == "-h" || a == "--help") {
            printf("lgchat [--host H] [--port P] [--temp T] [--max N] [--system S]\n"
                   "       [--no-think] [--hide-think] [--no-color] [-p PROMPT]\n"
                   "Interactive unless -p is given. /reset clears history, /quit exits.\n");
            return 0;
        }
    }
    if (!color) { DIM = PURPLE = RESET = BOLD = ""; }

    httplib::Client cli(host, port);
    cli.set_read_timeout(600);
    cli.set_write_timeout(600);

    { auto r = cli.Get("/healthz");
      if (!r || r->status != 200) {
          fprintf(stderr, "cannot reach server at %s:%d — is lgserve running?\n",
                  host.c_str(), port);
          return 1;
      } }

    json msgs = json::array();
    if (!system_prompt.empty())
        msgs.push_back(json{{"role", "system"}, {"content", system_prompt}});

    auto turn = [&](const std::string& user) {
        msgs.push_back(json{{"role", "user"}, {"content", user}});
        json body{{"messages", msgs}, {"stream", true}, {"temperature", temp},
                  {"max_tokens", max_tokens},
                  {"chat_template_kwargs", json{{"enable_thinking", thinking}}}};

        std::string content, reasoning, buf;
        bool in_think = false, printed_think_hdr = false;

        auto on_chunk = [&](const char* data, size_t len) {
            buf.append(data, len);
            size_t nl;
            while ((nl = buf.find("\n\n")) != std::string::npos) {
                std::string line = buf.substr(0, nl);
                buf.erase(0, nl + 2);
                if (line.rfind("data: ", 0) != 0) continue;
                std::string p = line.substr(6);
                if (p == "[DONE]") continue;
                json j;
                try { j = json::parse(p); } catch (...) { continue; }
                if (!j.contains("choices") || j["choices"].empty()) continue;
                const auto& d = j["choices"][0]["delta"];
                if (d.contains("reasoning_content") && d["reasoning_content"].is_string()) {
                    std::string t = d["reasoning_content"];
                    reasoning += t;
                    if (show_think) {
                        if (!printed_think_hdr) { printf("%s%s[thinking]%s%s ", PURPLE, BOLD, RESET, DIM);
                                                  printed_think_hdr = true; in_think = true; }
                        fputs(t.c_str(), stdout);
                    }
                }
                if (d.contains("content") && d["content"].is_string()) {
                    std::string t = d["content"];
                    if (in_think) { printf("%s\n\n", RESET); in_think = false; }
                    content += t;
                    fputs(t.c_str(), stdout);
                }
                if (d.contains("tool_calls")) {
                    if (in_think) { printf("%s\n\n", RESET); in_think = false; }
                    printf("\n%s[tool_calls]%s %s\n", BOLD, RESET, d["tool_calls"].dump(1).c_str());
                }
                fflush(stdout);
            }
            return true;
        };

        auto r = cli.Post("/v1/chat/completions", httplib::Headers{}, body.dump(),
                          "application/json", on_chunk);
        if (in_think) printf("%s", RESET);
        printf("\n");
        if (!r) { fprintf(stderr, "request failed\n"); msgs.erase(msgs.end() - 1); return; }
        json am{{"role", "assistant"}, {"content", content}};
        if (!reasoning.empty()) am["reasoning_content"] = reasoning;
        msgs.push_back(am);
    };

    if (!one_shot.empty()) { turn(one_shot); return 0; }

    printf("%sLaguna S 2.1%s  —  /reset to clear, /quit to exit\n", BOLD, RESET);
    std::string line;
    for (;;) {
        printf("%s>%s ", BOLD, RESET);
        fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        if (line == "/quit" || line == "/exit") break;
        if (line == "/reset") {
            msgs = json::array();
            if (!system_prompt.empty())
                msgs.push_back(json{{"role", "system"}, {"content", system_prompt}});
            printf("%s(history cleared)%s\n", DIM, RESET);
            continue;
        }
        if (line.empty()) continue;
        turn(line);
    }
    return 0;
}
