#include "../include/laguna_config.h"
#include <cstdio>
int main(int argc,char**argv){
    using namespace laguna;
    Config c = load_config(std::string(argc>1?argv[1]:"models/Laguna-S-2.1-NVFP4")+"/config.json");
    printf("{\"full\":{\"scale\":%.17g,\"inv\":[",c.rope_full.attention_factor);
    auto a=c.rope_full.inv_freq(c.head_dim);
    for(size_t i=0;i<a.size();++i) printf("%s%.9g",i?",":"",a[i]);
    printf("]},\"slide\":{\"scale\":%.17g,\"inv\":[",c.rope_slide.attention_factor);
    auto b=c.rope_slide.inv_freq(c.head_dim);
    for(size_t i=0;i<b.size();++i) printf("%s%.9g",i?",":"",b[i]);
    printf("]}}\n");
    return 0;
}
