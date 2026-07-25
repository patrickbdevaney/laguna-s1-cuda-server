// Gate S1b: the C++ renderer must reproduce the shipped chat_template.jinja byte-for-byte,
// and the parser must round-trip reasoning + tool calls.
#include "../include/chat.h"
#include <cstdio>
#include <fstream>
using namespace lgchat;
static int PASS=0, FAIL=0;
static void show(const std::string& s){ for(char c:s) putchar(c=='\n'?'|':c); }
int main(){
    std::ifstream f("docs/kernel_refs/chat_ref.json");
    if(!f){ printf("run oracle/dump_chat_ref.py first\n"); return 1; }
    nlohmann::ordered_json J; f>>J;
    for(auto& c : J){
        std::string name = c["name"].get<std::string>();
        const auto& kw = c["kw"];
        std::vector<Message> msgs;
        for(const auto& m : kw["messages"]){
            Message mm;
            mm.role = m.at("role").get<std::string>();
            mm.content = m.value("content","");
            mm.reasoning = m.value("reasoning","");
            if(m.contains("tool_calls"))
                for(const auto& t : m["tool_calls"]){
                    ToolCall tc; tc.name = t["function"]["name"].get<std::string>();
                    tc.arguments = t["function"]["arguments"].dump();
                    mm.tool_calls.push_back(tc);
                }
            msgs.push_back(mm);
        }
        lgchat::ojson tools = kw.contains("tools") ? kw["tools"] : lgchat::ojson::array();
        std::string got = render(msgs, tools, kw.value("enable_thinking",true),
                                 kw.value("add_generation_prompt",false));
        std::string want = c["rendered"].get<std::string>();
        if(got==want){ printf("  OK   %s\n",name.c_str()); ++PASS; }
        else{
            ++FAIL; printf("  FAIL %s\n",name.c_str());
            size_t d=0; while(d<got.size()&&d<want.size()&&got[d]==want[d]) ++d;
            printf("       first diff at byte %zu\n       want: ...",d);
            show(want.substr(d>30?d-30:0, 90)); printf("\n       got : ...");
            show(got.substr(d>30?d-30:0, 90)); printf("\n");
        }
    }
    // parser round-trip
    printf("\n-- parser --\n");
    {
        std::string out = "I should check.</think>Let me look."
                          "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
                          "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call></assistant>";
        Parsed p = parse_output(out, true);
        bool ok = p.reasoning=="I should check." && p.content=="Let me look."
                  && p.tool_calls.size()==1 && p.tool_calls[0].name=="get_weather";
        lgchat::ojson a = lgchat::ojson::parse(p.tool_calls[0].arguments);
        ok &= a["city"]=="Paris" && a["days"]==3 && a["days"].is_number();
        printf("  %-34s %s\n","reasoning+toolcall+arg types",ok?"OK":"FAIL"); ok?++PASS:++FAIL;
        if(!ok) printf("       reasoning=%s content=%s args=%s\n",
                       p.reasoning.c_str(),p.content.c_str(),p.tool_calls[0].arguments.c_str());
    }
    {
        Parsed p = parse_output("</think>plain answer</assistant>", false);
        bool ok = p.reasoning.empty() && p.content=="plain answer" && p.tool_calls.empty();
        printf("  %-34s %s\n","thinking off",ok?"OK":"FAIL"); ok?++PASS:++FAIL;
    }
    {   // a string that merely looks numeric must stay a string if it is not valid JSON
        Parsed p = parse_output("x</think>a<tool_call>f<arg_key>k</arg_key>"
                                "<arg_value>3 apples</arg_value></tool_call></assistant>", true);
        lgchat::ojson a = lgchat::ojson::parse(p.tool_calls[0].arguments);
        bool ok = a["k"].is_string() && a["k"]=="3 apples";
        printf("  %-34s %s\n","ambiguous arg stays a string",ok?"OK":"FAIL"); ok?++PASS:++FAIL;
    }
    printf("\nchat: %d passed, %d failed\n",PASS,FAIL);
    return FAIL?1:0;
}
