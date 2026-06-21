"""Visual theme for the Image-Talk Gradio app: CSS, theme-toggle JS, masthead."""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Crimson+Pro:ital,wght@0,300;0,400;0,600;1,300&display=swap');

:root {
    --bg-primary:#0c0c0f; --bg-secondary:#131318; --bg-tertiary:#1a1a22;
    --bg-elevated:#1f1f2a;
    --border-subtle:rgba(255,255,255,.06); --border-mid:rgba(255,255,255,.12);
    --border-strong:rgba(255,255,255,.22);
    --accent:#5eead4; --accent-dim:rgba(94,234,212,.15); --accent-glow:rgba(94,234,212,.35);
    --text-primary:#f0eff5; --text-secondary:#a8a7b8; --text-muted:#5a5970;
    --ok:#4ade80; --err:#f87171; --warn:#fbbf24;
    --radius-sm:3px; --radius-md:6px;
    --shadow-card:0 4px 24px rgba(0,0,0,.4);
    --transition:all .2s cubic-bezier(.4,0,.2,1);
}
[data-theme="light"] {
    --bg-primary:#f5f3ee; --bg-secondary:#ede9e0; --bg-tertiary:#e4dfd4;
    --bg-elevated:#fff;
    --border-subtle:rgba(0,0,0,.06); --border-mid:rgba(0,0,0,.12);
    --border-strong:rgba(0,0,0,.22);
    --accent:#0d9488; --accent-dim:rgba(13,148,136,.12); --accent-glow:rgba(13,148,136,.28);
    --text-primary:#1a1820; --text-secondary:#4a4860; --text-muted:#8a88a0;
    --ok:#16a34a; --err:#dc2626; --warn:#d97706;
    --shadow-card:0 4px 24px rgba(0,0,0,.07);
}
*,*::before,*::after{box-sizing:border-box;}
body,.gradio-container,.gradio-container *{font-family:'Crimson Pro',Georgia,serif!important;}
body,.gradio-container{background:var(--bg-primary)!important;color:var(--text-primary)!important;transition:var(--transition);}

/* Masthead */
.it-masthead{position:relative;background:var(--bg-secondary);border-bottom:1px solid var(--border-subtle);overflow:hidden;}
.masthead-inner{display:flex;align-items:stretch;min-height:104px;}
.masthead-brand{flex:1;padding:22px 36px;position:relative;}
.masthead-wordmark{font-family:'Bebas Neue',sans-serif!important;font-size:4.2rem;letter-spacing:.1em;line-height:.85;color:var(--text-primary)!important;text-shadow:0 0 80px var(--accent-glow);display:block;}
.masthead-wordmark span{color:var(--accent)!important;}
.masthead-tagline{font-family:'Bebas Neue',sans-serif!important;font-size:1rem;letter-spacing:.3em;text-transform:uppercase;color:var(--accent)!important;margin-top:4px;display:block;opacity:.85;}
.masthead-sub{font-family:'DM Mono',monospace!important;font-size:.58rem;letter-spacing:.22em;text-transform:uppercase;color:var(--text-muted)!important;margin-top:7px;display:block;}
.masthead-stats{width:320px;background:var(--bg-primary);border-left:1px solid var(--border-subtle);padding:16px 24px;display:flex;flex-direction:column;justify-content:space-between;}
.stat-row{display:flex;align-items:center;justify-content:space-between;padding:2px 0;}
.stat-label{font-family:'DM Mono',monospace!important;font-size:.55rem;letter-spacing:.14em;text-transform:uppercase;color:var(--text-muted)!important;}
.stat-val{font-family:'DM Mono',monospace!important;font-size:.62rem;color:var(--accent)!important;font-weight:500;}
.theme-row{display:flex;align-items:center;gap:8px;margin-top:6px;padding-top:8px;border-top:1px solid var(--border-subtle);}
.theme-row-label{font-family:'DM Mono',monospace!important;font-size:.55rem;letter-spacing:.14em;text-transform:uppercase;color:var(--text-muted)!important;}
.theme-toggle-btn{background:var(--bg-tertiary)!important;border:1px solid var(--border-mid)!important;border-radius:20px!important;padding:4px 14px!important;font-family:'DM Mono',monospace!important;font-size:.56rem!important;letter-spacing:.1em!important;text-transform:uppercase!important;color:var(--text-secondary)!important;cursor:pointer;transition:var(--transition)!important;}
.theme-toggle-btn:hover{border-color:var(--accent)!important;color:var(--accent)!important;background:var(--accent-dim)!important;}

/* Tabs */
.tab-nav{background:var(--bg-secondary)!important;border-bottom:1px solid var(--border-subtle)!important;padding:0 24px!important;}
.tab-nav button{font-family:'DM Mono',monospace!important;font-size:.6rem!important;letter-spacing:.18em!important;text-transform:uppercase!important;color:var(--text-muted)!important;background:transparent!important;border:none!important;border-bottom:2px solid transparent!important;padding:14px 18px!important;transition:var(--transition)!important;}
.tab-nav button.selected{color:var(--text-primary)!important;border-bottom-color:var(--accent)!important;}
.tab-nav button:hover{color:var(--text-secondary)!important;}

/* Cards */
.gr-panel,.gr-group,.gr-box,.block,.form{background:var(--bg-secondary)!important;border:1px solid var(--border-subtle)!important;border-radius:var(--radius-md)!important;box-shadow:var(--shadow-card)!important;transition:var(--transition)!important;}
.gr-panel:hover,.gr-group:hover{border-color:var(--border-mid)!important;}

/* Section labels */
.section-label{font-family:'DM Mono',monospace!important;font-size:.56rem!important;letter-spacing:.2em!important;text-transform:uppercase!important;color:var(--accent)!important;margin-bottom:14px!important;display:flex!important;align-items:center!important;gap:8px!important;}
.section-label::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--border-mid),transparent);}

/* Form */
label,.gr-label{font-family:'DM Mono',monospace!important;font-size:.6rem!important;letter-spacing:.12em!important;text-transform:uppercase!important;color:var(--text-muted)!important;}
textarea,input[type=text],input[type=number],.gr-input{background:var(--bg-tertiary)!important;border:1px solid var(--border-subtle)!important;border-radius:var(--radius-sm)!important;color:var(--text-primary)!important;font-family:'Crimson Pro',Georgia,serif!important;font-size:.95rem!important;line-height:1.6!important;padding:10px 14px!important;transition:var(--transition)!important;}
textarea:focus,input:focus{border-color:var(--accent)!important;box-shadow:0 0 0 3px var(--accent-dim)!important;outline:none!important;background:var(--bg-elevated)!important;}

/* Buttons */
.gr-button{font-family:'DM Mono',monospace!important;font-size:.65rem!important;letter-spacing:.18em!important;text-transform:uppercase!important;border-radius:var(--radius-sm)!important;transition:var(--transition)!important;}
button.primary,.gr-button.primary{background:var(--accent)!important;color:#06201c!important;border:none!important;font-weight:500!important;padding:12px 28px!important;letter-spacing:.2em!important;box-shadow:0 2px 12px var(--accent-glow)!important;}
button.primary:hover{filter:brightness(1.08)!important;box-shadow:0 4px 20px var(--accent-glow)!important;transform:translateY(-1px)!important;}
button.secondary,.gr-button.secondary{background:transparent!important;border:1px solid var(--border-mid)!important;color:var(--text-secondary)!important;}
button.secondary:hover{border-color:var(--accent)!important;color:var(--accent)!important;background:var(--accent-dim)!important;}
input[type=range]{accent-color:var(--accent)!important;height:3px!important;}
input[type=checkbox],input[type=radio]{accent-color:var(--accent)!important;}

/* Status */
.status-ok{font-family:'DM Mono',monospace;font-size:.72rem;color:var(--ok);letter-spacing:.05em;}
.status-err{font-family:'DM Mono',monospace;font-size:.72rem;color:var(--err);letter-spacing:.05em;}
.status-warn{font-family:'DM Mono',monospace;font-size:.72rem;color:var(--warn);letter-spacing:.05em;}
.audio-info{font-family:'DM Mono',monospace;font-size:.62rem;letter-spacing:.08em;padding:8px 12px;border-radius:var(--radius-sm);background:var(--bg-tertiary);border:1px solid var(--border-subtle);margin:4px 0;color:var(--text-secondary);}

/* Footer */
.vram-footer{background:var(--bg-secondary);border-top:1px solid var(--border-subtle);padding:10px 24px;display:flex;align-items:center;justify-content:space-between;}
.vram-text{font-family:'DM Mono',monospace;font-size:.58rem;letter-spacing:.12em;text-transform:uppercase;color:var(--text-muted);}
.vram-accent{color:var(--accent);}

/* Output media */
.output-media img,.output-media video{border-radius:var(--radius-md)!important;border:1px solid var(--border-subtle)!important;box-shadow:var(--shadow-card)!important;}

::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:var(--bg-primary)}
::-webkit-scrollbar-thumb{background:var(--border-mid);border-radius:2px}
::-webkit-scrollbar-thumb:hover{background:var(--accent)}
@keyframes fadeSlideUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
.gradio-container>*{animation:fadeSlideUp .4s cubic-bezier(.4,0,.2,1) both;}
"""

THEME_JS = """
function toggleTheme(){
  const root=document.documentElement;
  const btn=document.getElementById('it-theme-btn');
  if(root.getAttribute('data-theme')==='light'){
    root.removeAttribute('data-theme'); if(btn) btn.textContent='☀ Light Mode';
  } else {
    root.setAttribute('data-theme','light'); if(btn) btn.textContent='◑ Dark Mode';
  }
}
"""

MASTHEAD = """
<div class="it-masthead">
  <div class="masthead-inner">
    <div class="masthead-brand">
      <span class="masthead-wordmark">IMAGE·<span>TALK</span></span>
      <span class="masthead-tagline">AI Media Studio</span>
      <span class="masthead-sub">v1.1 &nbsp;·&nbsp; Talking Video · Relip · FLUX · Face Swap</span>
    </div>
    <div class="masthead-stats">
      <div class="stat-row"><span class="stat-label">Talking Head</span><span class="stat-val">SadTalker + XTTS</span></div>
      <div class="stat-row"><span class="stat-label">Lip Re-sync</span><span class="stat-val">LatentSync</span></div>
      <div class="stat-row"><span class="stat-label">Text → Image</span><span class="stat-val">FLUX.1</span></div>
      <div class="stat-row"><span class="stat-label">Face Swap</span><span class="stat-val">InsightFace</span></div>
      <div class="theme-row">
        <span class="theme-row-label">Theme</span>
        <button class="theme-toggle-btn" id="it-theme-btn" onclick="toggleTheme()">☀ Light Mode</button>
      </div>
    </div>
  </div>
</div>
"""
