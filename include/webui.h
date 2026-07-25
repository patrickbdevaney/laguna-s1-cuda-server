// webui.h — the built-in chat page, served at /. One file, no build step, no CDN.
#pragma once

inline const char* kWebUI = R"HTML(<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Laguna S 2.1</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--fg:#e6e8ee;--dim:#8b93a7;--acc:#6ea8fe;--think:#a58bff}
@media(prefers-color-scheme:light){:root{--bg:#f7f8fa;--panel:#fff;--line:#e2e5ec;--fg:#1a1d24;--dim:#5f6878;--acc:#2563eb;--think:#6d43d9}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:600}
.tag{font-size:12px;color:var(--dim)}
main{max-width:860px;margin:0 auto;padding:16px 16px 140px}
.msg{margin:0 0 14px;padding:11px 13px;border:1px solid var(--line);border-radius:10px;background:var(--panel);white-space:pre-wrap;word-wrap:break-word;overflow-wrap:anywhere}
.msg.user{background:transparent;border-color:transparent;padding-left:0;padding-right:0}
.who{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--dim);margin-bottom:5px}
details.think{margin:0 0 9px;border-left:2px solid var(--think);padding:2px 0 2px 10px;color:var(--dim);font-size:14px}
details.think summary{cursor:pointer;color:var(--think);font-size:12px;text-transform:uppercase;letter-spacing:.06em}
pre{background:rgba(127,127,127,.12);padding:10px;border-radius:8px;overflow-x:auto;margin:8px 0}
code{font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
footer{position:fixed;bottom:0;left:0;right:0;background:var(--bg);border-top:1px solid var(--line);padding:10px 16px}
.row{max-width:860px;margin:0 auto;display:flex;gap:8px;align-items:flex-end}
textarea{flex:1;min-height:44px;max-height:180px;resize:vertical;padding:11px;border-radius:9px;border:1px solid var(--line);background:var(--panel);color:var(--fg);font:inherit}
button{padding:11px 16px;border-radius:9px;border:1px solid var(--line);background:var(--acc);color:#fff;font:inherit;font-weight:600;cursor:pointer}
button.ghost{background:transparent;color:var(--dim)}
button:disabled{opacity:.5;cursor:default}
label{font-size:12px;color:var(--dim);display:flex;gap:5px;align-items:center}
input[type=number]{width:72px;padding:4px 6px;border-radius:6px;border:1px solid var(--line);background:var(--panel);color:var(--fg);font:inherit}
#stat{font-size:12px;color:var(--dim);max-width:860px;margin:6px auto 0}
</style></head><body>
<header>
  <h1>Laguna S 2.1</h1>
  <span class="tag">NVFP4 + DFlash &middot; pure CUDA</span>
  <label><input type="checkbox" id="think" checked> thinking</label>
  <label>temp <input type="number" id="temp" value="0" min="0" max="2" step="0.1"></label>
  <label>max <input type="number" id="maxtok" value="512" min="1" max="8192" step="64"></label>
  <button class="ghost" id="clear">clear</button>
</header>
<main id="log"></main>
<footer>
  <div class="row">
    <textarea id="in" placeholder="Message Laguna&hellip;  (Enter to send, Shift+Enter for newline)"></textarea>
    <button id="send">Send</button>
    <button class="ghost" id="stop" style="display:none">Stop</button>
  </div>
  <div id="stat"></div>
</footer>
<script>
const log=document.getElementById('log'), inp=document.getElementById('in');
const sendB=document.getElementById('send'), stopB=document.getElementById('stop');
let msgs=[], ctl=null;

function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function fmt(s){
  let out='', i=0;
  const re=/```([\w+-]*)\n([\s\S]*?)```/g; let m;
  while((m=re.exec(s))){ out+=esc(s.slice(i,m.index)); out+='<pre><code>'+esc(m[2])+'</code></pre>'; i=re.lastIndex; }
  out+=esc(s.slice(i)); return out;
}
function bubble(role){
  const d=document.createElement('div'); d.className='msg '+role;
  d.innerHTML='<div class="who">'+role+'</div><div class="think-slot"></div><div class="body"></div>';
  log.appendChild(d); window.scrollTo(0,document.body.scrollHeight); return d;
}
function setThink(el,txt){
  let s=el.querySelector('.think-slot');
  if(!txt){s.innerHTML='';return}
  if(!s.firstChild) s.innerHTML='<details class="think" open><summary>reasoning</summary><div></div></details>';
  s.querySelector('div').textContent=txt;
}
document.getElementById('clear').onclick=()=>{msgs=[];log.innerHTML='';document.getElementById('stat').textContent=''};
inp.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();go()}});
sendB.onclick=go;
stopB.onclick=()=>{if(ctl)ctl.abort()};

async function go(){
  const text=inp.value.trim(); if(!text||ctl) return;
  inp.value=''; msgs.push({role:'user',content:text});
  const ub=bubble('user'); ub.querySelector('.body').textContent=text;
  const ab=bubble('assistant'); const body=ab.querySelector('.body');
  let content='', reasoning='', n=0; const t0=performance.now();
  sendB.disabled=true; stopB.style.display='';
  ctl=new AbortController();
  try{
    const r=await fetch('/v1/chat/completions',{method:'POST',signal:ctl.signal,
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({messages:msgs,stream:true,
        temperature:parseFloat(document.getElementById('temp').value)||0,
        max_tokens:parseInt(document.getElementById('maxtok').value)||512,
        chat_template_kwargs:{enable_thinking:document.getElementById('think').checked}})});
    if(!r.ok){body.textContent='HTTP '+r.status+': '+await r.text(); return}
    const rd=r.body.getReader(), dec=new TextDecoder(); let buf='';
    for(;;){
      const {done,value}=await rd.read(); if(done) break;
      buf+=dec.decode(value,{stream:true});
      let nl;
      while((nl=buf.indexOf('\n\n'))>=0){
        const line=buf.slice(0,nl); buf=buf.slice(nl+2);
        if(!line.startsWith('data: ')) continue;
        const p=line.slice(6); if(p==='[DONE]') continue;
        let j; try{j=JSON.parse(p)}catch(e){continue}
        const d=j.choices&&j.choices[0]&&j.choices[0].delta; if(!d) continue;
        if(d.reasoning_content){reasoning+=d.reasoning_content; setThink(ab,reasoning); n++}
        if(d.content){content+=d.content; body.innerHTML=fmt(content); n++}
        if(d.tool_calls){content+='\n\n[tool_calls] '+JSON.stringify(d.tool_calls,null,1); body.innerHTML=fmt(content)}
        window.scrollTo(0,document.body.scrollHeight);
      }
    }
    msgs.push({role:'assistant',content:content,reasoning_content:reasoning});
    const dt=(performance.now()-t0)/1000;
    document.getElementById('stat').textContent=n+' chunks in '+dt.toFixed(2)+'s';
  }catch(e){ if(e.name!=='AbortError') body.textContent+='\n[error] '+e.message; }
  finally{ ctl=null; sendB.disabled=false; stopB.style.display='none'; }
}
</script></body></html>
)HTML";
