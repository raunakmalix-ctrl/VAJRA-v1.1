"""Visual theme for VAJRA (Gradio): CSS, theme-toggle JS, masthead.

Cosmetic only — no component IDs or structure depend on this file.
Palette: white canvas · navy command panels · amber accent (military / HUD).
"""
import base64
import os

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _data_uri(name):
    """Base64-embed a small local asset so it renders with no extra Gradio
    static-file route (works identically in Colab and locally)."""
    with open(os.path.join(_ASSETS, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


_IITI_LOGO = _data_uri("iiti_logo.png")
_MCTE_LOGO = _data_uri("mcte_flash_logo.png")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600&family=Crimson+Pro:ital,wght@1,400;1,500&family=Share+Tech+Mono&display=swap');
@import url('https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3/dist/tabler-icons.min.css');

:root{
  --bg:#eef1f6; --bg-soft:#e6ebf2; --card:#ffffff;
  --navy:#173a5e; --navy-2:#1f4d7d; --navy-deep:#0f2740;
  --amber:#f5a623; --amber-deep:#dd8709; --amber-soft:rgba(245,166,35,.15);
  --ink:#15202e; --muted:#5c6b7e; --tagline:#98a2b3;
  --border:#d7dee7; --border-strong:#c1cad6;
  --ok:#1f9d55; --err:#d64545; --warn:#dd8709;
  --radius:6px; --radius-lg:10px;
  --shadow:0 8px 24px rgba(14,37,64,.10);
  --shadow-sm:0 2px 8px rgba(14,37,64,.08);
  --transition:all .18s cubic-bezier(.4,0,.2,1);
  --mono:'Share Tech Mono',ui-monospace,monospace;
  --grid-line:rgba(19,61,102,.05);
}
:root[data-theme="dark"]{
  --bg:#0a1420; --bg-soft:#0c1a29; --card:#102135;
  --navy:#143a60; --navy-2:#1d5085; --navy-deep:#070f19;
  --ink:#e9eff6; --muted:#93a1b3; --tagline:#7f8ba0;
  --border:#1f3550; --border-strong:#2c445f;
  --shadow:0 10px 30px rgba(0,0,0,.55); --shadow-sm:0 2px 10px rgba(0,0,0,.4);
  --grid-line:rgba(245,166,35,.06);
}

*,*::before,*::after{box-sizing:border-box;}
body,.gradio-container{
  background-color:var(--bg)!important; color:var(--ink)!important;
  font-family:'Inter',system-ui,sans-serif!important; transition:var(--transition);
}
/* faint site-wide tactical lattice (three sets of parallel lines = hex grid).
   Scoped to a single element with default (scroll) attachment -- `fixed`
   attachment plus layered repeating-gradients on two nested elements was
   expensive enough to stall rendering. */
.gradio-container{
  max-width:100%!important;
  background-image:
    repeating-linear-gradient(0deg, var(--grid-line) 0 1px, transparent 1px 42px),
    repeating-linear-gradient(60deg, var(--grid-line) 0 1px, transparent 1px 42px),
    repeating-linear-gradient(120deg, var(--grid-line) 0 1px, transparent 1px 42px);
}

/* ── Masthead ─────────────────────────────────────────────────────────── */
.vajra-masthead{
  position:relative; background:var(--card);
  border-bottom:1px solid var(--border);
  box-shadow:var(--shadow-sm); overflow:hidden;
}
.vajra-masthead::before{  /* faint tactical grid */
  content:''; position:absolute; inset:0; pointer-events:none; opacity:.5;
  background-image:linear-gradient(var(--border) 1px,transparent 1px),
    linear-gradient(90deg,var(--border) 1px,transparent 1px);
  background-size:44px 44px; mask-image:linear-gradient(90deg,transparent,#000 40%,transparent);
}
.vajra-masthead::after{  /* scanline sweep -- transform-only so it's compositor-driven
  and never forces layout, unlike animating `top` */
  content:''; position:absolute; left:0; right:0; height:2px; top:0;
  background:linear-gradient(90deg,transparent,rgba(245,166,35,.6),transparent);
  animation:vjScan 6s linear infinite; pointer-events:none; z-index:4;
  will-change:transform;
}
@keyframes vjScan{
  0%{transform:translateY(0); opacity:0;}
  6%{opacity:1;} 94%{opacity:1;}
  100%{transform:translateY(420px); opacity:0;}
}
.vj-rail{position:absolute; left:0; top:0; bottom:0; width:10px;
  background:linear-gradient(180deg,var(--amber),var(--amber-deep)); z-index:2;}

/* ── Institutional letterhead strip ───────────────────────────────────── */
.vj-letterhead{position:relative; z-index:3; display:flex; align-items:center;
  justify-content:space-between; gap:16px; padding:9px 40px 9px 56px;
  background:var(--navy-deep); border-bottom:1px solid rgba(245,166,35,.25);}
.vj-inst{display:flex; align-items:center; gap:10px; min-width:0;}
.vj-inst-right{flex-direction:row-reverse; text-align:right;}
.vj-inst-plate{flex:0 0 auto; width:34px; height:34px; border-radius:5px;
  background:#fff; display:flex; align-items:center; justify-content:center;
  box-shadow:0 1px 4px rgba(0,0,0,.35); overflow:hidden;}
.vj-inst-plate img{width:100%; height:100%; object-fit:contain; padding:2px;}
.vj-inst-text{font-family:'Rajdhani',sans-serif; font-weight:600; font-size:.62rem;
  letter-spacing:.07em; text-transform:uppercase; color:#9db4d0; line-height:1.32;}
:root[data-theme="dark"] .vj-inst-text{color:#7f93ab;}
.vj-letterhead-mid{font-family:var(--mono); font-size:.68rem; letter-spacing:.08em;
  text-transform:uppercase; color:var(--amber); display:flex; align-items:center;
  gap:8px; flex:0 0 auto;}
.vj-blip{width:6px; height:6px; border-radius:50%; background:var(--amber);
  box-shadow:0 0 0 3px rgba(245,166,35,.2); animation:vjPulseAmber 1.8s ease-out infinite;}
@keyframes vjPulseAmber{
  0%{box-shadow:0 0 0 0 rgba(245,166,35,.5);}
  70%{box-shadow:0 0 0 6px rgba(245,166,35,0);}
  100%{box-shadow:0 0 0 0 rgba(245,166,35,0);}
}

.vj-inner{position:relative; z-index:3; display:flex; align-items:center;
  justify-content:space-between; gap:28px; padding:26px 40px 24px 56px; flex-wrap:wrap;}
.vj-brand{position:relative;}
.vj-wordmark{
  font-family:'Oswald',sans-serif; font-weight:700;
  font-size:clamp(3rem,7vw,5.4rem); line-height:.9; letter-spacing:.16em;
  color:var(--amber); text-shadow:2px 3px 0 rgba(15,39,64,.16); display:block;
  cursor:default;
}
.vj-wordmark:hover{animation:vjGlitch .4s steps(2,end) 2;}
@keyframes vjGlitch{
  0%,100%{text-shadow:2px 3px 0 rgba(15,39,64,.16); transform:translate(0,0);}
  20%{text-shadow:-2px 0 #e0453f,2px 0 #2ad1ff,2px 3px 0 rgba(15,39,64,.16); transform:translate(-1px,0);}
  40%{text-shadow:2px 0 #e0453f,-2px 0 #2ad1ff,2px 3px 0 rgba(15,39,64,.16); transform:translate(1px,0);}
  60%{text-shadow:-1px 0 #e0453f,1px 0 #2ad1ff,2px 3px 0 rgba(15,39,64,.16); transform:translate(-1px,0);}
}
.vj-brand::after{content:''; display:block; width:118px; height:4px; margin-top:10px;
  background:linear-gradient(90deg,var(--amber),transparent);}
.vj-tagline{font-family:'Crimson Pro',Georgia,serif; font-style:italic;
  font-size:1.18rem; color:var(--tagline); margin-top:12px; display:block;}
.vj-subtitle{font-family:'Rajdhani',sans-serif; font-weight:600; font-size:.9rem;
  letter-spacing:.04em; text-transform:uppercase; color:var(--amber-deep);
  margin-top:8px; max-width:620px; display:block; line-height:1.45;}

.vj-stats{display:grid; grid-template-columns:1fr 1fr; gap:8px; align-content:center;}
.vj-chip{background:var(--navy); color:var(--amber);
  font-family:'Rajdhani',sans-serif; font-weight:600; font-size:.72rem;
  letter-spacing:.09em; text-transform:uppercase; text-align:center;
  padding:11px 18px; border-radius:8px; border:1px solid rgba(255,255,255,.07);
  box-shadow:0 3px 10px rgba(14,37,64,.20); transition:var(--transition);}
.vj-chip:hover{background:var(--navy-2); transform:translateY(-1px);}
.vj-toggle-wrap{grid-column:1/-1; display:flex; justify-content:flex-end; margin-top:2px;}
.vj-toggle{background:transparent; border:1.5px solid var(--navy);
  color:var(--navy); font-family:'Rajdhani',sans-serif; font-weight:600;
  font-size:.62rem; letter-spacing:.12em; text-transform:uppercase;
  padding:5px 14px; border-radius:20px; cursor:pointer; transition:var(--transition);}
.vj-toggle:hover{background:var(--navy); color:#fff;}
:root[data-theme="dark"] .vj-toggle{border-color:var(--amber); color:var(--amber);}

/* ── Tabs (navy command bar, amber active) ────────────────────────────── */
.tab-nav{background:var(--navy-deep)!important; border:none!important;
  border-bottom:3px solid var(--amber)!important; padding:0 16px!important;
  display:flex!important; flex-wrap:wrap!important; gap:2px!important;}
/* Unselected tabs sit on a permanently dark (navy-deep) bar in BOTH themes,
   so their text/icon color is a fixed gold rather than a theme variable --
   it needs to read against the same dark background either way. */
.tab-nav button{font-family:'Rajdhani',sans-serif!important; font-weight:600!important;
  font-size:.82rem!important; letter-spacing:.09em!important; text-transform:uppercase!important;
  color:var(--amber-deep)!important; background:transparent!important; border:none!important;
  border-bottom:3px solid transparent!important; margin-bottom:-3px!important;
  padding:13px 17px!important; transition:var(--transition)!important;}
.tab-nav button:hover{color:var(--amber)!important; background:rgba(255,255,255,.06)!important;}
.tab-nav button.selected{color:var(--navy-deep)!important; background:var(--amber)!important;
  border-bottom-color:var(--amber)!important; font-weight:700!important;}

/* ── Cards / panels ───────────────────────────────────────────────────── */
.gr-panel,.gr-group,.gr-box,.block,.form,.gr-accordion{
  background:var(--card)!important; border:1px solid var(--border)!important;
  border-radius:var(--radius-lg)!important; box-shadow:var(--shadow-sm)!important;
  transition:var(--transition)!important; position:relative!important;}
.gr-group:hover,.gr-panel:hover{border-color:var(--border-strong)!important;}

/* targeting-reticle corner brackets, revealed on hover (decorative only —
   pseudo-elements, so they never touch any component's actual content/color) */
.gr-panel::before,.gr-panel::after,.gr-group::before,.gr-group::after,
.gr-box::before,.gr-box::after,.block::before,.block::after,
.gr-accordion::before,.gr-accordion::after{
  content:''; position:absolute; width:10px; height:10px; border:2px solid var(--amber);
  opacity:0; transition:opacity .2s ease; pointer-events:none; z-index:5;}
.gr-panel::before,.gr-group::before,.gr-box::before,.block::before,.gr-accordion::before{
  top:-1px; left:-1px; border-right:none; border-bottom:none;}
.gr-panel::after,.gr-group::after,.gr-box::after,.block::after,.gr-accordion::after{
  bottom:-1px; right:-1px; border-left:none; border-top:none;}
.gr-panel:hover::before,.gr-panel:hover::after,.gr-group:hover::before,.gr-group:hover::after,
.gr-box:hover::before,.gr-box:hover::after,.block:hover::before,.block:hover::after,
.gr-accordion:hover::before,.gr-accordion:hover::after{opacity:1;}

/* Accordion headers (Media Studio) */
.gr-accordion .label-wrap,.gr-accordion span.label-wrap{
  font-family:'Rajdhani',sans-serif!important; font-weight:600!important;
  text-transform:uppercase!important; letter-spacing:.06em!important;
  color:var(--navy)!important; font-size:.9rem!important;}
:root[data-theme="dark"] .gr-accordion .label-wrap{color:var(--amber)!important;}

/* ── Section labels (custom) ──────────────────────────────────────────── */
.section-label{font-family:'Rajdhani',sans-serif!important; font-weight:700!important;
  font-size:.68rem!important; letter-spacing:.16em!important; text-transform:uppercase!important;
  color:var(--navy)!important; margin-bottom:14px!important;
  display:flex!important; align-items:center!important; gap:9px!important;}
.section-label::before{content:''; width:16px; height:3px; background:var(--amber); flex:0 0 auto;}
.section-label::after{content:''; flex:1; height:1px;
  background:linear-gradient(90deg,var(--border-strong),transparent);}
:root[data-theme="dark"] .section-label{color:var(--amber)!important;}

/* ── Form controls ────────────────────────────────────────────────────── */
label,.gr-label{font-family:'Rajdhani',sans-serif!important; font-weight:600!important;
  font-size:.72rem!important; letter-spacing:.08em!important; text-transform:uppercase!important;
  color:var(--muted)!important;}
textarea,input[type=text],input[type=number],select,.gr-input,.gr-text-input{
  background:var(--bg-soft)!important; border:1px solid var(--border-strong)!important;
  border-radius:var(--radius)!important; color:var(--ink)!important;
  font-family:'Inter',sans-serif!important; font-size:.94rem!important;
  padding:10px 13px!important; transition:var(--transition)!important;}
textarea:focus,input:focus,select:focus{
  border-color:var(--amber)!important; background:var(--card)!important;
  box-shadow:0 0 0 3px var(--amber-soft)!important; outline:none!important;}
input[type=range]{accent-color:var(--amber)!important; height:4px!important;}
input[type=checkbox],input[type=radio]{accent-color:var(--amber)!important; width:16px; height:16px;}

/* Radio/checkbox CHOICE text (e.g. "Wan2.2-I2V (best identity/multi-subject)")
   is interactive selection, not a secondary field caption -- it needs full
   contrast, unlike a field's own muted caption label. :has() targets any
   <label> wrapping a radio/checkbox input regardless of Gradio's internal
   wrapper class names, which differ across versions -- this was rendering
   as --muted (barely readable grey) in dark mode before this rule. */
label:has(input[type=radio]),label:has(input[type=checkbox]){
  color:var(--ink)!important;}

/* ── Buttons ──────────────────────────────────────────────────────────── */
.gr-button{font-family:'Rajdhani',sans-serif!important; font-weight:700!important;
  font-size:.78rem!important; letter-spacing:.11em!important; text-transform:uppercase!important;
  border-radius:var(--radius)!important; transition:var(--transition)!important;
  position:relative!important;}
button.primary,.gr-button.primary{
  background:var(--amber)!important; color:var(--navy-deep)!important; border:none!important;
  padding:12px 26px!important; box-shadow:0 4px 14px rgba(245,166,35,.35)!important;}
button.primary:hover{filter:brightness(1.06)!important; transform:translateY(-1px)!important;
  box-shadow:0 7px 20px rgba(245,166,35,.45)!important;}
button.primary:active{transform:translateY(0)!important;}
/* targeting-lock flash on hover — pure decoration, no layout impact */
button.primary::before,button.primary::after{
  content:''; position:absolute; width:9px; height:9px; border:2px solid var(--amber);
  opacity:0; transition:opacity .15s ease,transform .15s ease; pointer-events:none;}
button.primary::before{top:-6px; left:-6px; border-right:none; border-bottom:none; transform:translate(4px,4px);}
button.primary::after{bottom:-6px; right:-6px; border-left:none; border-top:none; transform:translate(-4px,-4px);}
button.primary:hover::before,button.primary:hover::after{opacity:1; transform:translate(0,0);}
button.secondary,.gr-button.secondary{
  background:transparent!important; color:var(--navy)!important;
  border:1.5px solid var(--navy)!important;}
button.secondary:hover{background:var(--navy)!important; color:#fff!important;}
:root[data-theme="dark"] button.secondary{color:var(--amber)!important; border-color:var(--amber)!important;}

/* ── Status / info ────────────────────────────────────────────────────── */
.status-ok{font-family:'Rajdhani',sans-serif; font-weight:600; color:var(--ok); letter-spacing:.05em; text-transform:uppercase; font-size:.8rem;}
.status-err{font-family:'Rajdhani',sans-serif; font-weight:600; color:var(--err); letter-spacing:.05em; text-transform:uppercase; font-size:.8rem;}
.status-warn{font-family:'Rajdhani',sans-serif; font-weight:600; color:var(--warn); letter-spacing:.05em; text-transform:uppercase; font-size:.8rem;}
.audio-info{font-family:'Rajdhani',sans-serif; font-weight:600; font-size:.72rem;
  letter-spacing:.06em; padding:8px 13px; border-radius:var(--radius);
  background:var(--bg-soft); border:1px solid var(--border); border-left:3px solid var(--amber);
  color:var(--muted); margin:4px 0;}

/* ── Output media ─────────────────────────────────────────────────────── */
.output-media img,.output-media video{border-radius:var(--radius)!important;
  border:1px solid var(--border)!important; box-shadow:var(--shadow)!important;}

/* ── Footer ───────────────────────────────────────────────────────────── */
.vram-footer{background:var(--navy-deep); border-top:2px solid var(--amber);
  padding:10px 26px; display:flex; align-items:center; justify-content:space-between;
  flex-wrap:wrap; gap:10px;}
.vram-text{font-family:'Rajdhani',sans-serif; font-weight:600; font-size:.66rem;
  letter-spacing:.13em; text-transform:uppercase; color:#9db4d0;}
.vram-accent{color:var(--amber); font-family:var(--mono);}
.vj-clock{font-family:var(--mono)!important; color:#9db4d0;}

::-webkit-scrollbar{width:9px; height:9px;}
::-webkit-scrollbar-track{background:var(--bg-soft);}
::-webkit-scrollbar-thumb{background:var(--border-strong); border-radius:5px;}
::-webkit-scrollbar-thumb:hover{background:var(--amber);}

@keyframes fadeSlideUp{from{opacity:0; transform:translateY(10px);} to{opacity:1; transform:translateY(0);}}
.gradio-container>*{animation:fadeSlideUp .35s cubic-bezier(.4,0,.2,1) both;}

/* ── Masthead logo mark ───────────────────────────────────────────────── */
.vj-brandrow{display:flex; align-items:center; gap:18px;}
.vj-logo{width:64px; height:64px; flex:0 0 auto;
  filter:drop-shadow(0 4px 10px rgba(14,37,64,.25));}

/* ── Tab icons (injected by JS) ───────────────────────────────────────── */
.tab-nav button i.ti{font-size:1rem; margin-right:8px; vertical-align:-2px;}

/* ── Status ribbon (under the tab bar) ────────────────────────────────── */
.status-ribbon{display:flex; align-items:center; gap:18px; flex-wrap:wrap;
  background:var(--card); border:1px solid var(--border);
  border-left:4px solid var(--amber); border-radius:var(--radius);
  padding:8px 16px; margin:12px 4px 0;}
.status-ribbon .sr-item{font-family:'Rajdhani',sans-serif; font-weight:600;
  font-size:.72rem; letter-spacing:.1em; text-transform:uppercase;
  color:var(--muted); display:flex; align-items:center; gap:7px;}
.status-ribbon .sr-item i{font-size:.95rem; color:var(--amber);}
.status-ribbon .sr-item b{color:var(--navy); font-weight:700;}
:root[data-theme="dark"] .status-ribbon .sr-item b{color:var(--ink);}
.sr-dot{width:8px; height:8px; border-radius:50%; background:var(--ok);
  box-shadow:0 0 0 3px rgba(31,157,85,.18); animation:vjPulse 1.8s ease-out infinite;}
.sr-live{color:var(--ok)!important;}
@keyframes vjPulse{
  0%{box-shadow:0 0 0 0 rgba(31,157,85,.5);}
  70%{box-shadow:0 0 0 7px rgba(31,157,85,0);}
  100%{box-shadow:0 0 0 0 rgba(31,157,85,0);}
}

/* ── Per-tab hero header ──────────────────────────────────────────────── */
.tab-hero{display:flex; align-items:center; gap:14px; padding:6px 2px 16px;
  border-bottom:1px solid var(--border); margin-bottom:18px;}
.tab-hero .th-ico{width:42px; height:42px; flex:0 0 auto; border-radius:9px;
  background:var(--navy); color:var(--amber); display:flex; align-items:center;
  justify-content:center; font-size:1.35rem; box-shadow:var(--shadow-sm);}
.tab-hero .th-txt{display:flex; flex-direction:column; gap:1px;}
.tab-hero .th-title{font-family:'Oswald',sans-serif; font-weight:600;
  font-size:1.12rem; letter-spacing:.04em; color:var(--navy); text-transform:uppercase;}
:root[data-theme="dark"] .tab-hero .th-title{color:var(--amber);}
.tab-hero .th-sub{font-family:'Inter',sans-serif; font-size:.86rem; color:var(--muted);}

/* ── Empty-state for output media ─────────────────────────────────────── */
.output-media .empty,.output-media .wrap{min-height:210px!important;}
.empty-hint{font-family:'Rajdhani',sans-serif; font-weight:600; font-size:.82rem;
  letter-spacing:.09em; text-transform:uppercase; color:var(--muted);
  display:flex; align-items:center; gap:8px;}
.empty-hint i{font-size:1rem; color:var(--amber);}

/* ── Readability fixes ─────────────────────────────────────────────────── */
/* Gradio's own component chrome (upload dropzones, hint text) defaults to a
   low-contrast grey that's unreadable on our white cards in light mode. This
   is deliberately LOW specificity (no !important, no wildcard class hacks) so
   it only fills in where Gradio has no more-specific rule of its own — e.g.
   it must NOT touch Radio/Dropdown pill internals, which rely on their own
   selected/unselected color rules to render at all. */
.gradio-container p, .gradio-container span, .gradio-container div,
.gradio-container li, .gradio-container td, .gradio-container small{
  color:var(--ink);
}
"""

THEME_JS = """
function vajraToggle(){
  const root=document.documentElement;
  const btn=document.getElementById('vajra-theme-btn');
  if(root.getAttribute('data-theme')==='dark'){
    root.removeAttribute('data-theme'); if(btn) btn.textContent='◐ Dark Mode';
  } else {
    root.setAttribute('data-theme','dark'); if(btn) btn.textContent='☀ Light Mode';
  }
}
// Inject Tabler icons into the tab buttons (Gradio renders tab labels as text).
const VAJRA_TAB_ICONS=['ti-user-video','ti-pencil','ti-photo','ti-mask',
  'ti-movie','ti-adjustments','ti-wand','ti-user-star'];
function vajraTabIcons(){
  const btns=document.querySelectorAll('.tab-nav button');
  btns.forEach((b,i)=>{
    if(b.querySelector('i.ti')||!VAJRA_TAB_ICONS[i]) return;
    const ic=document.createElement('i');
    ic.className='ti '+VAJRA_TAB_ICONS[i];
    b.insertBefore(ic,b.firstChild);
  });
}
new MutationObserver(vajraTabIcons).observe(document.documentElement,{childList:true,subtree:true});
setTimeout(vajraTabIcons,600); setTimeout(vajraTabIcons,1800);

// Live UTC clock in the footer ("SYS TIME" readout).
function vajraClock(){
  const el=document.getElementById('vajra-clock');
  if(!el) return;
  const d=new Date();
  const p=n=>String(n).padStart(2,'0');
  el.textContent='SYS TIME '+p(d.getUTCHours())+':'+p(d.getUTCMinutes())+':'+p(d.getUTCSeconds())+'Z';
}
setInterval(vajraClock,1000); vajraClock();
"""

MASTHEAD = """
<div class="vajra-masthead">
  <div class="vj-rail"></div>
  <div class="vj-letterhead">
    <div class="vj-inst">
      <span class="vj-inst-plate"><img src="__IITI_LOGO__" alt="IIT Indore"/></span>
      <span class="vj-inst-text">Indian Institute of<br/>Technology Indore</span>
    </div>
    <div class="vj-letterhead-mid"><span class="vj-blip"></span>Secure Local Compute Node</div>
    <div class="vj-inst vj-inst-right">
      <span class="vj-inst-text">Military College of<br/>Telecommunication Engineering</span>
      <span class="vj-inst-plate"><img src="__MCTE_LOGO__" alt="MCTE"/></span>
    </div>
  </div>
  <div class="vj-inner">
    <div class="vj-brand">
      <div class="vj-brandrow">
        <svg class="vj-logo" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
          <rect width="64" height="64" rx="13" fill="#0f2740"/>
          <path d="M37 7 L17 35 H29 L25 57 L47 25 H34 Z" fill="#f5a623"/>
        </svg>
        <span class="vj-wordmark">VAJRA</span>
      </div>
      <span class="vj-tagline">Digital Lies. Kinetic Chaos.</span>
      <span class="vj-subtitle">Unified, Offline, Multi-Modal Deep Learning Platform
        for Image, Voice &amp; Video Synthesis</span>
    </div>
    <div class="vj-stats">
      <div class="vj-chip">Image Diffusion</div>
      <div class="vj-chip">Face Swap</div>
      <div class="vj-chip">Voice Clone</div>
      <div class="vj-chip">Video Relip</div>
      <div class="vj-chip">Talking Face</div>
      <div class="vj-chip">Avatar Studio</div>
      <div class="vj-toggle-wrap">
        <button class="vj-toggle" id="vajra-theme-btn" onclick="(function(b){var r=document.documentElement;if(r.getAttribute('data-theme')==='dark'){r.removeAttribute('data-theme');b.textContent='◐ Dark Mode';}else{r.setAttribute('data-theme','dark');b.textContent='☀ Light Mode';}})(this)">◐ Dark Mode</button>
      </div>
    </div>
  </div>
</div>
"""
MASTHEAD = MASTHEAD.replace("__IITI_LOGO__", _IITI_LOGO).replace("__MCTE_LOGO__", _MCTE_LOGO)
