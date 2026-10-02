// Chequeo de regresión del visor viz/crossrun/ (Node, sin dependencias).
//
// Ejecuta el <script> COMPLETO de index.html contra un DOM falso y recorre todo:
// 2 sets x 30 pares ordenados x 2 niveles x 2 colores de banda, los tooltips de
// cada banda y cada nodo, el panel de foco de cada nodo, y la matriz en sus 4
// medidas x 2 tratamientos del -1. Falla si hay una excepción o si algún HTML
// emitido contiene "undefined" o "NaN".
//
// Existe porque la primera versión solo probaba la geometría del Sankey
// (computeLayout) y nunca llamaba al render: un bug que rompía la página entera
// (PAY.noise_pid mal ruteado al set activo) pasó todas las verificaciones.
//
// Uso, desde la raíz del repo:
//     python scripts/viz/build_crossrun.py
//     node scripts/viz/check_crossrun_viewer.js [ruta/a/crossrun.json]
const fs = require('fs');
const html = fs.readFileSync('viz/crossrun/index.html', 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const payloadText = fs.readFileSync(process.argv[2] || 'viz/crossrun/crossrun.json', 'utf8');

// ---- DOM mínimo ----
class El {
  constructor(tag) { this.tagName = (tag||'div').toLowerCase(); this.children = []; this.attrs = {};
    this.style = {}; this.dataset = {}; this._cls = new Set(); this.hidden = false;
    this._text = ''; this._html = ''; this.value = ''; this.disabled = false; this.listeners = {}; }
  get classList() { const s = this._cls; return {
    add: (c) => s.add(c), remove: (c) => s.delete(c), contains: (c) => s.has(c),
    toggle: (c, f) => { const on = f === undefined ? !s.has(c) : !!f; on ? s.add(c) : s.delete(c); return on; } }; }
  set className(v) { this._cls = new Set(String(v).split(/\s+/).filter(Boolean)); }
  get className() { return [...this._cls].join(' '); }
  set textContent(v) { this._text = String(v); if (v === '') this.children = []; }
  get textContent() { return this._text; }
  set innerHTML(v) { this._html = String(v); this.children = []; }
  get innerHTML() { return this._html; }
  setAttribute(k, v) { this.attrs[k] = String(v);
    if (k === 'class') this.className = v;
    if (k.startsWith('data-')) this.dataset[k.slice(5).replace(/-(\w)/g, (_, c) => c.toUpperCase())] = String(v); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  appendChild(c) { this.children.push(c); c.parentNode = this; return c; }
  addEventListener(ev, fn) { (this.listeners[ev] ||= []).push(fn); }
  getBoundingClientRect() { return { width: 200, height: 80, left: 0, top: 0 }; }
  get clientWidth() { return 1100; }
  *walk() { for (const c of this.children) { yield c; yield* c.walk(); } }
  querySelectorAll(sel) {
    const parts = sel.split(',').map((s) => s.trim());
    const match = (e, s) => {
      if (s.startsWith('[')) { const k = s.slice(1, -1).replace(/^data-/, '').replace(/-(\w)/g, (_, c) => c.toUpperCase()); return k in e.dataset; }
      const [tag, cls] = s.split('.');
      return (!tag || e.tagName === tag) && (!cls || e._cls.has(cls));
    };
    return [...this.walk()].filter((e) => parts.some((s) => match(e, s)));
  }
}
const byId = new Map();
const doc = {
  documentElement: new El('html'),
  querySelector(sel) { if (!byId.has(sel)) byId.set(sel, new El('div')); return byId.get(sel); },
  querySelectorAll() { return []; },
  createElement: (t) => new El(t),
  createElementNS: (_, t) => new El(t),
  createTextNode: (s) => { const e = new El('#text'); e.textContent = s; return e; },
};
const errors = [];
const sandbox = {
  document: doc, location: { hash: '' }, history: { replaceState() {} },
  addEventListener() {}, innerWidth: 1400, innerHeight: 900,
  setTimeout, clearTimeout, requestAnimationFrame: (f) => f(),
  console: { ...console, error: (...a) => { errors.push(a.map(String).join(' ')); } },
  fetch: () => Promise.resolve({ ok: true, json: () => Promise.resolve(JSON.parse(payloadText)) }),
};
const run = new Function(...Object.keys(sandbox), script + `
  ;return { state, render, renderMatrix, bandTip, nodeTip, showPanel, applyFocus,
            get LAYOUT() { return LAYOUT; }, get PAY() { return PAY; }, S };`);
const P = run(...Object.values(sandbox));

(async () => {
  await new Promise((r) => setTimeout(r, 20));      // deja resolver el fetch
  const status = doc.querySelector('#status').textContent;
  if (/error|no se pudo|formato/.test(status) || errors.length) {
    console.log('FALLA EN LA CARGA INICIAL:', status, '\n', errors.join('\n')); process.exit(1);
  }
  let views = 0, tips = 0, panels = 0, cells = 0;
  for (const set of Object.keys(P.PAY.sets)) {
    P.state.set = set;
    const R = P.S().runs.length;
    for (const level of ['proceso', 'cluster']) {
      P.state.level = level; P.state.view = 'flujo';
      for (let a = 0; a < R; a++) for (let b = 0; b < R; b++) {
        if (a === b) continue;
        P.state.a = a; P.state.b = b; P.state.focus = null;
        for (const colorBy of ['origen', 'destino']) {
          P.state.colorBy = colorBy; P.state.diff = colorBy === 'destino';
          P.render(); views++;
        }
        const L = P.LAYOUT;
        // tooltips de todas las bandas y todos los nodos
        for (const bd of L.bands) { const h = P.bandTip(bd); if (/undefined|NaN/.test(h)) throw new Error(`bandTip ${set}/${level}/${a}->${b}: ${h}`); tips++; }
        for (const [side, order] of [['A', L.orderA], ['B', L.orderB]]) for (const id of order) {
          const h = P.nodeTip(side, id); if (/undefined|NaN/.test(h)) throw new Error(`nodeTip ${set}/${level}/${a}->${b} ${side}:${id}: ${h}`); tips++;
          P.state.focus = { side, id }; P.applyFocus(); P.showPanel(); panels++;
          if (/undefined|NaN/.test(doc.querySelector('#panel').innerHTML)) throw new Error(`panel ${side}:${id}`);
        }
        P.state.focus = null;
      }
    }
    P.state.view = 'matriz';
    for (const metric of ['proc', 'ariproc', 'aricl', 'nest']) for (const noise of ['excluir', 'cat11']) {
      P.state.metric = metric; P.state.noise = noise; P.render(); cells += R * (R - 1);
      if (/undefined|NaN/.test(doc.querySelector('#matrixNote').innerHTML)) throw new Error(`matrixNote ${metric}/${noise}`);
    }
  }
  if (errors.length) { console.log('console.error durante el recorrido:\n' + errors.join('\n')); process.exit(1); }
  console.log(`OK — script completo de la página, sin excepciones ni undefined/NaN en el HTML emitido`);
  console.log(`  vistas de aluvial renderizadas: ${views}`);
  console.log(`  tooltips generados:             ${tips.toLocaleString('es-AR')}`);
  console.log(`  paneles de foco abiertos:       ${panels.toLocaleString('es-AR')}`);
  console.log(`  celdas de matriz renderizadas:  ${cells}  (2 sets x 4 medidas x 2 tratamientos del -1)`);
})().catch((e) => { console.log('FALLA:', e.message); process.exit(1); });
