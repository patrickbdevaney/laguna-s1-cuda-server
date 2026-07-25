// Gate S1a: the C++ ByteLevel BPE must reproduce the HF tokenizer exactly.
#include "../include/tokenizer.h"
#include "../include/third_party/json.hpp"
#include <cstdio>
#include <fstream>
int main(int argc,char**argv){
    std::string md = argc>1?argv[1]:"models/Laguna-S-2.1-NVFP4";
    lgtok::Tokenizer tk(md+"/tokenizer.json");
    printf("vocab %zu\n", tk.vocab_size());
    std::ifstream f("docs/kernel_refs/tok_ref.json");
    if(!f){ printf("run oracle/dump_tok_ref.py first\n"); return 1; }
    nlohmann::json J; f>>J;
    int pass=0, fail=0;
    for(auto& c : J){
        std::string text = c["text"].get<std::string>();
        std::vector<int> want; for(auto&x:c["ids"]) want.push_back(x.get<int>());
        std::vector<int> got = tk.encode(text);
        std::string dec = tk.decode(got);
        bool ok = (got==want), rt = (dec==text);
        if(ok&&rt) ++pass; else {
            ++fail;
            std::string disp = text.size()>44?text.substr(0,44)+"...":text;
            for(auto&ch:disp) if(ch=='\n') ch='|'; else if(ch=='\t') ch='>';
            printf("  FAIL %-48s\n", disp.c_str());
            printf("       want(%zu):", want.size());
            for(size_t i=0;i<want.size()&&i<16;++i) printf(" %d",want[i]); printf("\n");
            printf("       got (%zu):", got.size());
            for(size_t i=0;i<got.size()&&i<16;++i) printf(" %d",got[i]); printf("\n");
            if(!rt) printf("       roundtrip FAILED\n");
        }
    }
    printf("\ntokenizer: %d passed, %d failed\n", pass, fail);
    return fail?1:0;
}
