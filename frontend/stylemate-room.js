/* ===================================================================
   StyleMate - 3D fitting room, backend client and model settings.
   Loaded as a module AFTER the page's classic scripts, so the handful of
   globals it reassigns (openFitRoom, runAgent, llmModal) override the
   originals while every other page function keeps working untouched.
   =================================================================== */
import * as THREE from './three.module.min.js';
import { OrbitControls } from './OrbitControls.js';

/* ---------- backend client ---------- */
const API = (() => {
  try { return localStorage.getItem('sm_api') || ''; } catch (e) { return ''; }
})() || (location.protocol.startsWith('http') ? location.origin : 'http://localhost:8000');

const SM = {
  base: API,
  async json(path, body, opts = {}) {
    const r = await fetch(this.base + path, {
      method: body === undefined ? 'GET' : 'POST',
      headers: body instanceof FormData ? undefined : { 'Content-Type': 'application/json' },
      body: body instanceof FormData ? body : (body === undefined ? undefined : JSON.stringify(body)),
      ...opts
    });
    const text = await r.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch (e) { /* non-JSON error body */ }
    if (!r.ok) {
      const msg = (data && (data.detail || data.message)) || text.slice(0, 200) || ('HTTP ' + r.status);
      throw new Error(msg);
    }
    return data;
  },
  health() { return this.json('/api/health'); },
  settings() { return this.json('/api/settings/llm'); },
  saveSettings(cfg) { return this.json('/api/settings/llm', cfg); },
  geminiModels() { return this.json('/api/settings/gemini/models'); },
  sync(products) { return this.json('/api/catalogue/sync', { products }); },
  search(q, limit, profile) { return this.json('/api/search', { q, limit, profile, useModel: true }); },
  look(file, audience) {
    const fd = new FormData();
    fd.append('photo', file); fd.append('audience', audience || '');
    fd.append('perItem', '6'); fd.append('useVision', '1');
    return this.json('/api/look', fd);
  },
  avatarPhoto(file) {
    const fd = new FormData(); fd.append('photo', file);
    return this.json('/api/avatar/photo', fd);
  },
  forgetPhoto(id) {
    const fd = new FormData(); fd.append('photoId', id);
    return this.json('/api/avatar/forget', fd);
  },
  fittingRoom(profile, items, photoId) {
    return this.json('/api/fittingroom', { profile, items, photoId });
  }
};
window.SM = SM;


/* Page globals are declared with `let`, which puts them in the global lexical
   environment rather than on `window`, so `window.all` is undefined. Indirect
   eval runs in global scope and can read them. */
const G = name => { try { return (0, eval)(name); } catch (e) { return undefined; } };
const Gall = () => G('all') || [];
const Gfit = () => G('fit') || { queue: [] };
const Gcart = () => G('cart');
const Gsize = () => G('sizeProfile') || {};
const Gcall = (name, ...args) => { const f = G(name); return typeof f === 'function' ? f(...args) : undefined; };

/* Who the shopper is shopping for: 'women', 'men' or null (both). From v4 the
   page's Colour Studio sets it (shopFor). An older page has no shopFor, so its
   size-profile gender is used, as before. */
function shopperAudience(){
  const f = G('shopFor');
  if (typeof f === 'function') return f();
  const g = Gsize().gender;
  return g === 'female' ? 'women' : g === 'male' ? 'men' : null;
}

/* Body measurements in the shape the backend expects. sizeProfile stores cm
   internally, so unit is always cm regardless of what the UI is showing. */
function bodyProfile() {
  const p = Gsize();
  const num = v => (typeof v === 'number' && isFinite(v) && v > 0) ? v : null;
  return {
    unit: 'cm', gender: p.gender || 'male',
    chest: num(p.chest), waist: num(p.waist), hips: num(p.hips),
    height: num(p.height), foot: num(p.foot), shoulder: num(p.shoulder),
    frontLength: num(p.frontLength), fitPref: p.fitPref || 'true'
  };
}
window.bodyProfile = bodyProfile;

/* ---------- 3D viewer ---------- */
class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf2efe8);
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.05, 60);
    this.camera.position.set(0.0, 1.05, 3.1);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0.95, 0);
    this.controls.minDistance = 0.9;
    this.controls.maxDistance = 7;
    this.lights();
    this.body = null;
    this.garments = new THREE.Group();
    this.scene.add(this.garments);
    this.running = true;
    this._loop = this._loop.bind(this);
    this._loop();
    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(canvas.parentElement || canvas);
    this.resize();
  }
  lights() {
    /* Three-point studio rig: key, fill, rim, plus a soft hemisphere so the
       silhouette never goes flat. */
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xd8d2c4, 0.72));
    const key = new THREE.DirectionalLight(0xfff6e8, 1.5);
    key.position.set(2.4, 3.4, 2.8); key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 0.5; key.shadow.camera.far = 12;
    key.shadow.camera.left = -2; key.shadow.camera.right = 2;
    key.shadow.camera.top = 3; key.shadow.camera.bottom = -0.5;
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xe8eeff, 0.55);
    fill.position.set(-2.8, 1.6, 1.8); this.scene.add(fill);
    const rim = new THREE.DirectionalLight(0xffffff, 0.8);
    rim.position.set(-0.6, 2.4, -3.2); this.scene.add(rim);

    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(3.2, 64),
      new THREE.MeshStandardMaterial({ color: 0xe6e1d6, roughness: 0.95, metalness: 0 }));
    floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true;
    this.scene.add(floor);
    const grid = new THREE.GridHelper(6, 24, 0xcdc6b8, 0xdfd9cc);
    grid.position.y = 0.001; this.scene.add(grid);
  }
  resize() {
    const el = this.canvas.parentElement || this.canvas;
    const w = Math.max(240, el.clientWidth), h = Math.max(240, el.clientHeight);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }
  _loop() {
    if (!this.running) return;
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame(this._loop);
  }
  dispose() {
    this.running = false;
    try { this._ro.disconnect(); } catch (e) {}
    this.scene.traverse(o => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => m.dispose());
    });
    this.renderer.dispose();
  }
  static geom(positions, indices) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    g.setIndex(indices);
    g.computeVertexNormals();
    return g;
  }
  setBody(mesh, skinRgb) {
    if (this.body) { this.scene.remove(this.body); this.body.geometry.dispose(); this.body.material.dispose(); }
    const col = skinRgb ? new THREE.Color(skinRgb[0] / 255, skinRgb[1] / 255, skinRgb[2] / 255)
                        : new THREE.Color(0xd8b79a);
    const mat = new THREE.MeshStandardMaterial({ color: col, roughness: 0.78, metalness: 0.02, flatShading: false });
    this.body = new THREE.Mesh(Viewer.geom(mesh.positions, mesh.indices), mat);
    this.body.castShadow = true; this.body.receiveShadow = true;
    this.scene.add(this.body);
    this.controls.target.set(0, mesh.heightM * 0.55, 0);
    this.frame(mesh.heightM);
  }
  frame(heightM) {
    const h = heightM || 1.7;
    this.camera.position.set(0, h * 0.62, h * 1.85);
    this.controls.target.set(0, h * 0.55, 0);
    this.controls.update();
  }
  clearGarments() {
    [...this.garments.children].forEach(c => {
      this.garments.remove(c);
      if (c.geometry) c.geometry.dispose();
      if (c.material) c.material.dispose();
    });
  }
  addGarment(g, colourHex, visible) {
    if (!g.positions || !g.positions.length) return null;
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(colourHex), roughness: 0.86, metalness: 0.02,
      side: THREE.DoubleSide, transparent: true, opacity: 0.94
    });
    const m = new THREE.Mesh(Viewer.geom(g.positions, g.indices), mat);
    m.castShadow = true; m.visible = visible !== false;
    m.userData.sku = g.sku;
    this.garments.add(m);
    return m;
  }
  highlight(sku) {
    this.garments.children.forEach(m => {
      m.material.emissive = new THREE.Color(m.userData.sku === sku ? 0x332211 : 0x000000);
      m.material.needsUpdate = true;
    });
  }
}

/* Catalogue colour name -> a usable hex for the shell material. */
const COLOUR_HEX = {
  'black': 0x242424, 'white': 0xf2f2ee, 'off white': 0xece7da, 'grey': 0x8d8d8d,
  'charcoal': 0x4a4e52, 'navy blue': 0x1e2c58, 'blue': 0x3161b4, 'teal': 0x247a7a,
  'green': 0x3a7a42, 'olive': 0x6f7040, 'red': 0xb02a2a, 'maroon': 0x6d1e2a,
  'pink': 0xde88a0, 'purple': 0x6c4696, 'lavender': 0xbeb0dc, 'yellow': 0xe2c448,
  'mustard': 0xc49a30, 'orange': 0xd67630, 'rust': 0xa25030, 'brown': 0x704e38,
  'coffee brown': 0x523a2c, 'beige': 0xd6c4a6, 'cream': 0xeee2c6, 'tan': 0xc49e6e,
  'khaki': 0xb2a276, 'gold': 0xc6a248, 'silver': 0xbec2c6, 'peach': 0xf0ba9e,
  'magenta': 0xba3482, 'turquoise blue': 0x3ebac2, 'sea green': 0x2e8b74,
  'burgundy': 0x6b1f2e, 'mauve': 0xb08aa0, 'rose': 0xd98fa0, 'copper': 0xa8643c,
  'steel': 0x8894a0, 'grey melange': 0x9a9a95, 'multi': 0x9a7f6a
};
const hexFor = name => COLOUR_HEX[(name || '').toLowerCase().trim()] ?? 0x9a8f80;

export { SM, Viewer, hexFor, bodyProfile };

/* ===================================================================
   FITTING ROOM UI
   =================================================================== */
const ROOM = { viewer:null, data:null, photoId:null, photoName:'', sizes:{}, hidden:{}, busy:false, status:'' };
window.ROOM = ROOM;

const esc2 = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const $$ = id => document.getElementById(id);

function queuedProducts(){
  const q = Gfit().queue || [];
  return q.map(id => Gall().find(p => p.id === id)).filter(Boolean);
}

function measurementSummary(){
  const p = bodyProfile();
  const show = (k,label) => p[k] ? `<span><b>${label}</b> ${Math.round(p[k])} cm</span>` : '';
  const have = [['chest','Chest'],['waist','Waist'],['hips','Hips'],['shoulder','Shoulder'],
                ['frontLength','Front length'],['height','Height'],['foot','Foot']]
    .map(([k,l]) => show(k,l)).filter(Boolean);
  return have.length ? have.join('') :
    '<span class="room-warn">No measurements saved yet &mdash; open My Size first.</span>';
}

function roomHTML(){
  const q = queuedProducts(), p = bodyProfile();
  return `
  <button class="close-tone" data-roomclose="1" aria-label="Close">&times;</button>
  <div class="room-grid">
    <div class="room-stage">
      <div class="room-canvas-wrap"><canvas id="room-canvas"></canvas>
        <div class="room-overlay" id="room-overlay"${ROOM.data ? ' style="display:none"' : ''}>${ROOM.busy ? 'Building your avatar&hellip;' : 'Creating your avatar&hellip;'}</div>
        ${q.length ? '' : '<div class="room-emptytag">No pieces yet &middot; add one from the store to try it on</div>'}
        <div class="room-viewbtns">
          <button data-roomview="front">Front</button><button data-roomview="side">Side</button>
          <button data-roomview="back">Back</button><button data-roomview="reset">Reset view</button>
        </div>
      </div>
      <div class="room-disclose" id="room-disclose">Drag to rotate, scroll to zoom.</div>
    </div>
    <div class="room-side">
      <div class="kicker" style="color:var(--rust)">STYLEMATE &middot; FITTING ROOM</div>
      <h3 style="margin:6px 0 2px">Create my <em>3D avatar</em></h3>
      ${q.length ? '' : `<div class="room-empty"><b>Nothing to try on yet.</b> The fitting room needs at least one piece:
        use <b>＋ Add to Fitting Room</b> on any product, then open the fitting room again. Your avatar is built
        from your measurements either way.
        <button class="abtn" data-roomclose="1" type="button">Pick something to try on &#8599;</button></div>`}
      <section class="room-sec">
        <div class="alabel" style="margin-top:0">A &middot; MY AVATAR</div>
        <div class="room-gender"><span>Body model</span>
          <button class="room-chip ${p.gender==='male'?'on':''}" data-roomgender="male">Male</button>
          <button class="room-chip ${p.gender==='female'?'on':''}" data-roomgender="female">Female</button>
        </div>
        <div class="room-meas">${measurementSummary()}</div>
        <div class="room-row">
          <button class="room-ghost" data-roompick="1">Upload full-body photo</button>
          <input type="file" id="room-photo" accept="image/jpeg,image/png,image/webp" hidden>
          ${ROOM.photoId ? '<button class="room-ghost" data-roomforget="1">Forget my photo</button>' : ''}
        </div>
        <div class="room-photoname">${ROOM.photoName ? 'Using: '+esc2(ROOM.photoName) : 'Optional. Measurements alone build a valid avatar.'}</div>
        <button class="abtn" data-roombuild="1" style="margin-top:9px">${ROOM.data?'Rebuild avatar':'Create my 3D avatar'} &#8599;</button>
        <div class="room-status" id="room-status">${esc2(ROOM.status)}</div>
      </section>
      <section class="room-sec">
        <div class="alabel">C &middot; ADD A PIECE &middot; ${q.length}</div>
        ${q.length ? '<div class="room-pieces">' + q.map(pr => {
          const sz = ROOM.sizes[pr.id] || '', off = ROOM.hidden[pr.id];
          return `<div class="room-piece${off?' off':''}">
            ${thumb(pr.id, 'piece-thumb')}
            <div class="room-piece-b"><b>${esc2(pr.name)}</b>
              <span>${esc2(pr.articleType||pr.brand||'')}${pr.colour?' &middot; '+esc2(pr.colour):''}</span>
              <div class="room-piece-ctl">
                <select data-roomsize="${esc2(pr.id)}"><option value="">Best size</option>
                  ${['XS','S','M','L','XL','XXL','UK 6','UK 7','UK 8','UK 9','UK 10']
                    .map(z=>`<option value="${z}"${sz===z?' selected':''}>${z}</option>`).join('')}</select>
                <button data-roomtoggle="${esc2(pr.id)}">${off?'Show':'Hide'}</button>
                <button data-roomremove="${esc2(pr.id)}">Remove</button>
              </div></div></div>`;
        }).join('') + '</div>'
        : '<div class="adisc">Nothing added yet. Use <b>Add to Fitting Room</b> on any product card.</div>'}
      </section>
      <section class="room-sec" id="room-fit">
        <div class="alabel">D &middot; FIT ANALYSIS</div>
        <div class="adisc">Create the avatar to run the measurement comparison.</div>
      </section>
      <section class="room-sec">
        <div class="alabel">E &middot; WHEN YOU ARE HAPPY</div>
        <div class="adisc">Review the fit report above before buying.</div>
        <button class="abtn" data-roomcart="1" style="margin-top:8px">Add reviewed items to cart &#8599;</button>
      </section>
    </div>
  </div>`;
}

function setStatus(msg, kind){
  ROOM.status = msg || '';
  const el = $$('room-status');
  if (el){ el.textContent = msg || ''; el.className = 'room-status' + (kind ? ' '+kind : ''); }
}

/* Put the avatar and its garments back into the scene from the data we
   already hold. Used after any redraw, with no server round-trip. */
function restoreScene(){
  if (!ROOM.data || !ROOM.data.mesh) return;
  const canvas = $$('room-canvas');
  if (!canvas) return;
  if (!ROOM.viewer || ROOM.viewer.canvas !== canvas){
    if (ROOM.viewer) ROOM.viewer.dispose();
    ROOM.viewer = new Viewer(canvas);
  }
  const ov = $$('room-overlay'); if (ov) ov.style.display = 'none';
  ROOM.viewer.setBody(ROOM.data.mesh, ROOM.data.skinRgb);
  ROOM.viewer.clearGarments();
  (ROOM.data.garments || []).forEach(g =>
    ROOM.viewer.addGarment(g, hexFor(g.colour), !ROOM.hidden[g.sku]));
}

function redrawRoom(){
  const host = $$('room-card');
  if (!host) return;
  host.innerHTML = roomHTML();
  wireRoom();
  renderFitReports();
  // innerHTML replaced the <canvas>, so the old renderer is now drawing into
  // a detached element and the stage goes blank. Rebind to the new one.
  requestAnimationFrame(restoreScene);
}

function renderFitReports(){
  const host = $$('room-fit');
  if (!host) return;
  const reports = (ROOM.data && ROOM.data.fitReports) || [];
  const garments = (ROOM.data && ROOM.data.garments) || [];
  if (!reports.length){
    host.innerHTML = '<div class="alabel">D &middot; FIT ANALYSIS</div><div class="adisc">Add a piece, then create the avatar.</div>';
    return;
  }
  const badge = v => `<span class="fit-badge f-${String(v).replace(/\s+/g,'-')}">${esc2(v)}</span>`;
  host.innerHTML = '<div class="alabel">D &middot; FIT ANALYSIS</div>' + reports.map(r => {
    const g = garments.find(x => x.sku === r.sku) || {};
    const dims = (r.dims||[]).filter(d => d.verdict !== 'unknown');
    return `<div class="fit-card2">
      <div class="fit-head"><b>${esc2(g.name || r.sku)}</b> ${badge(r.overall)}</div>
      <div class="fit-sub">Size ${esc2(r.size)} &middot; confidence ${esc2(r.confidence)} &middot; ${r.chart_source==='supplier'?'supplier chart':'reference chart'}</div>
      ${dims.length ? `<table class="fit-tbl"><thead><tr><th>Where</th><th>Garment</th><th>You</th><th>Ease</th><th></th></tr></thead><tbody>${
        dims.map(d=>`<tr><td>${esc2(d.dim)}</td><td>${d.garment_cm??'-'}</td><td>${d.body_cm??'-'}</td><td>${d.ease_cm===null?'-':(d.ease_cm>0?'+':'')+d.ease_cm}</td><td>${badge(d.verdict)}</td></tr>`).join('')
      }</tbody></table>` : ''}
      <div class="fit-why">${(r.dims||[]).map(d=>esc2(d.note)).join('<br>')}</div>
      ${r.recommended_size && r.recommended_size !== r.size
        ? `<div class="fit-rec">Consider <b>${esc2(r.recommended_size)}</b> &mdash; ${esc2(r.size_change_reason||'')}</div>` : ''}
      ${(r.missing_body||[]).length || (r.missing_garment||[]).length ? `<div class="fit-missing">
        ${(r.missing_body||[]).length?'Not checked, because you have not saved: <b>'+esc2(r.missing_body.join(', '))+'</b>. ':''}
        ${(r.missing_garment||[]).length?'The chart omits: <b>'+esc2(r.missing_garment.join(', '))+'</b>.':''}</div>` : ''}
      <div class="fit-conf">${esc2(r.confidence_reason||'')}</div>
      <div class="fit-geo">${esc2(g.note||'')}</div>
      <button class="room-ghost" data-roomcartone="${esc2(r.sku)}">Add this to cart</button>
    </div>`;
  }).join('') + `<div class="fit-foot">${esc2((ROOM.data && ROOM.data.disclosure)||'')}</div>`;
}

async function buildAvatar(){
  if (ROOM.busy) return;
  ROOM.busy = true;
  setStatus('Building your avatar on the server...', 'busy');
  const items = queuedProducts().map(p => ({
    sku: p.id, size: ROOM.sizes[p.id] || null,
    product: { id:p.id, sourceSku:p.sourceSku, name:p.name, brand:p.brand,
               articleType:p.articleType, masterCategory:p.masterCategory,
               cat:p.cat, colour:p.colour, spec:p.spec, image:p.image }
  }));
  try{
    const data = await SM.fittingRoom(bodyProfile(), items, ROOM.photoId);
    ROOM.data = data;
    miniSetBody(data.mesh);
    restoreScene();
    const drawn = (data.garments||[]).filter(g=>g.assetKind==='measurement_shell').length;
    const skipped = (data.garments||[]).filter(g=>g.assetKind==='none').length;
    let msg = `Avatar built from ${data.supplied.length} saved measurement${data.supplied.length===1?'':'s'}`;
    if (data.defaulted.length) msg += `; ${data.defaulted.length} used a population default`;
    msg += `. ${drawn} garment shape${drawn===1?'':'s'} placed.`;
    if (skipped) msg += ` ${skipped} item${skipped===1?' has':'s have'} nothing to draw on the body.`;
    setStatus(msg, 'ok');
    const d = $$('room-disclose');
    if (d) d.innerHTML = esc2(data.disclosure) + ((data.notes||[]).length ? '<br>'+data.notes.map(esc2).join('<br>') : '');
    renderFitReports();
  } catch(e){
    setStatus('Could not build the avatar: ' + e.message + '  Is the StyleMate backend running on ' + SM.base + ' ?', 'err');
  } finally { ROOM.busy = false; }
}

async function uploadRoomPhoto(file){
  if (!file) return;
  if (file.size > 12*1024*1024){ setStatus('That photo is over 12 MB.', 'err'); return; }
  setStatus('Analysing your photo...', 'busy');
  try{
    const r = await SM.avatarPhoto(file);
    ROOM.photoId = r.photoId; ROOM.photoName = file.name;
    const ev = r.evidence || {};
    const note = 'Photo analysed (' + (ev.source||'none') + ', confidence ' + (ev.confidence||'none') + '). ' + (ev.notes||[]).join(' ');
    redrawRoom(); setStatus(note, 'ok');
    await buildAvatar();
  } catch(e){ setStatus('Photo upload failed: ' + e.message, 'err'); }
}

function openFitRoom3D(){
  const card = $$('fit-card') || $$('room-card');
  const modal = $$('fit-modal');
  if (!card || !modal) return;
  card.id = 'room-card'; card.classList.add('room-card');
  card.innerHTML = roomHTML(); modal.hidden = false; wireRoom();
  if (!ROOM.data){
    // Build on open. The avatar is the feature; an empty canvas is not a
    // useful first impression.
    requestAnimationFrame(() => buildAvatar());
    return;
  }
  if (ROOM.data && ROOM.data.mesh){
    requestAnimationFrame(() => { restoreScene(); renderFitReports(); });
  }
}

function closeFitRoom3D(){
  const modal = $$('fit-modal'); if (modal) modal.hidden = true;
  if (ROOM.viewer){ ROOM.viewer.dispose(); ROOM.viewer = null; }
}

function wireRoom(){
  const f = $$('room-photo');
  if (f) f.addEventListener('change', e => uploadRoomPhoto(e.target.files && e.target.files[0]));
}

document.addEventListener('click', async e => {
  const b = e.target.closest('button'); if (!b) return;
  const d = b.dataset || {};
  if (d.roomclose){ closeFitRoom3D(); return; }
  if (d.roompick){ const f = $$('room-photo'); if (f) f.click(); return; }
  if (d.roombuild){ buildAvatar(); return; }
  if (d.roomforget){
    if (ROOM.photoId){ try{ await SM.forgetPhoto(ROOM.photoId); }catch(err){} }
    ROOM.photoId = null; ROOM.photoName = '';
    redrawRoom(); setStatus('Photo removed from the server.', 'ok'); return;
  }
  if (d.roomgender){
    const sp = Gsize(); if (sp) sp.gender = d.roomgender;
    Gcall('saveSizeProfile');
    redrawRoom(); setStatus('Switching to the ' + d.roomgender + ' body model...', 'busy');
    buildAvatar(); return;
  }
  if (d.roomtoggle){
    ROOM.hidden[d.roomtoggle] = !ROOM.hidden[d.roomtoggle];
    if (ROOM.viewer) ROOM.viewer.garments.children.forEach(m => {
      if (m.userData.sku === d.roomtoggle) m.visible = !ROOM.hidden[d.roomtoggle];
    });
    redrawRoom(); return;
  }
  if (d.roomremove){
    (0, eval)('fit.queue = fit.queue.filter(function(x){return x!==' +
      JSON.stringify(d.roomremove) + '})');
    Gcall('saveFitQueue');
    delete ROOM.sizes[d.roomremove]; delete ROOM.hidden[d.roomremove];
    redrawRoom();
    Gcall('render');
    buildAvatar(); return;
  }
  if (d.roomview && ROOM.viewer){
    const v = ROOM.viewer, h = (ROOM.data && ROOM.data.mesh.heightM) || 1.7;
    if (d.roomview === 'front') v.camera.position.set(0, h*0.62, h*1.85);
    if (d.roomview === 'side')  v.camera.position.set(h*1.85, h*0.62, 0);
    if (d.roomview === 'back')  v.camera.position.set(0, h*0.62, -h*1.85);
    if (d.roomview === 'reset') v.frame(h);
    v.controls.target.set(0, h*0.55, 0); v.controls.update(); return;
  }
  if (d.roomcartone){
    const c = Gcart();
    if (c && !c.includes(d.roomcartone)){
      c.push(d.roomcartone);
      Gcall('saveCart'); Gcall('render');
      Gcall('toast', 'Added to cart after fit review');
    }
    return;
  }
  if (d.roomcart){
    const q = queuedProducts().filter(p => !ROOM.hidden[p.id]);
    let n = 0;
    const c = Gcart() || [];
    q.forEach(p => { if (!c.includes(p.id)){ c.push(p.id); n++; } });
    Gcall('saveCart'); Gcall('render');
    Gcall('toast', n ? n+' item'+(n===1?'':'s')+' added to cart' : 'Already in your cart');
    return;
  }
});

document.addEventListener('change', e => {
  const s = e.target;
  if (s && s.dataset && s.dataset.roomsize){
    ROOM.sizes[s.dataset.roomsize] = s.value || '';
    buildAvatar();
  }
});

/* A product added from the catalogue while the room is open should appear on
   the avatar without another click. */
window.roomRefreshQueue = function(){
  if (!document.getElementById('room-card')) return;
  redrawRoom();
  buildAvatar();
};

window.openFitRoom = openFitRoom3D;
window.closeFitRoom = closeFitRoom3D;

/* ===================================================================
   SEARCH, SHOP A LOOK AND MODEL SETTINGS -> backend
   =================================================================== */
let CATALOGUE_SYNCED = false;

async function syncCatalogue(force){
  if (CATALOGUE_SYNCED && !force) return true;
  const rows = Gall().map(p => ({
    id:p.id, sourceSku:p.sourceSku, cat:p.cat, name:p.name, brand:p.brand,
    articleType:p.articleType, masterCategory:p.masterCategory,
    subCategory:p.subCategory, usage:p.usage, season:p.season, year:p.year,
    colour:p.colour, audience:p.audience, spec:p.spec, size:p.size,
    price:p.price, image:p.image || p.photo
  }));
  if (!rows.length) return false;
  try{
    for (let i = 0; i < rows.length; i += 400){
      await SM.sync(rows.slice(i, i + 400));
    }
    CATALOGUE_SYNCED = true;
    return true;
  } catch(e){ return false; }
}
window.syncCatalogue = syncCatalogue;

/* The page's own agent remains the offline path. When the backend answers we
   use it, because that is where the model understanding lives. */
const localRunAgent = window.runAgent;

async function backendSearch(q){
  const ok = await syncCatalogue();
  if (!ok) throw new Error('Could not sync the catalogue to the backend.');
  // Include the saved gender so the backend can shop for this person by
  // default when the query does not name an audience.
  const aud = shopperAudience();
  const prof = Object.assign({}, Gcall('styleProfile') || {},
                             { gender: aud === 'women' ? 'female' : aud === 'men' ? 'male' : null });
  return SM.search(q, 36, prof);
}

window.runAgent = async function(q){
  const query = (q || (document.getElementById('agent-prompt') || {}).value || '').trim();
  if (!query) return;
  const resp = document.getElementById('agent-response');
  // The page keeps this panel hidden until its own agent runs, so writing into
  // it was invisible. Unhide it whenever we put something there.
  // Write into a box of our own: the panel also holds the page's #agent-trace
  // and #wardrobe-flag, and replacing its whole content deleted them - so the
  // wardrobe check never showed and the in-page agent then crashed.
  const setMsg = h => {
    if (!resp) return;
    let box = document.getElementById('agent-backend');
    if (!box){ box = document.createElement('div'); box.id = 'agent-backend'; resp.prepend(box); }
    box.innerHTML = h; resp.hidden = false;
  };
  setMsg('<div class="adisc">Asking the StyleMate agent&hellip;</div>');
  try{
    const r = await backendSearch(query);
    const ids = new Set(r.results.map(x => x.id));
    const picked = Gall().filter(p => ids.has(p.id));
    const order = new Map(r.results.map((x, i) => [x.id, i]));
    picked.sort((a, b) => (order.get(a.id) ?? 99) - (order.get(b.id) ?? 99));
    const whyById = new Map(r.results.map(x => [x.id, x.why || []]));
    const s = r.spec || {};
    const chips = (r.filtersApplied || []).map(f => '<span class="spec-chip">' + f + '</span>').join('');
    // The answer goes in first, so the page can put its wardrobe note inside it.
    setMsg(
      (r.answer ? '<div class="agent-answer' + (r.answerFromModel ? '' : ' rules') + '">' +
        '<span class="agent-answer-tag">' +
        (r.answerFromModel ? 'STYLEMATE AGENT' : 'STYLEMATE &middot; NO MODEL CONNECTED') +
        '</span>' + esc2(r.answer) + '</div>' : '') +
      ((r.results || []).length ? ''
        : '<div class="adisc">Nothing matched. Try widening the budget or occasion.</div>')
      /* "What the agent understood" is switched off: shoppers do not see it.
         To bring it back (for debugging), add this to the message above:
      + '<div class="alabel">WHAT THE AGENT UNDERSTOOD</div>' +
      '<div class="spec-chips">' + chips + '</div>' +
      '<div class="agent-expl">' + (r.explanation || '') + '</div>' +
      (s.source === 'model'
        ? '<div class="adisc">Read by the language model.</div>'
        : '<div class="adisc">Read by rule parsing' +
          ((s.notes || []).length ? ' &mdash; ' + s.notes[0] : '') + '</div>') +
      ((r.results || []).length
        ? '<div class="agent-why">' + r.results.slice(0, 4).map(x =>
            '<div><b>' + x.name + '</b> &mdash; ' + (x.why || []).join('; ') + '</div>').join('') + '</div>'
        : '')
      */
    );
    // Hand the ranked list to the page, which shows it the way its own agent
    // does: wardrobe check, similar-to-owned tags and the styling panel on the
    // right. _relevance keeps the backend's order under "Recommended". Pages
    // before v11 have no showSearchResults and get the grid only, as before.
    if (typeof G('showSearchResults') === 'function'){
      const ranked = picked.map((p, i) => Object.assign({}, p, { _relevance: picked.length - i }));
      Gcall('showSearchResults', ranked, '\u201c' + query + '\u201d');
    } else {
      (0, eval)('agentResults = ' + JSON.stringify(picked.map(p => p.id)) +
        '.map(function(i){return all.find(function(p){return p.id===i})}).filter(Boolean)');
      Gcall('render');
    }
    if (typeof window.scrollTo === 'function') window.scrollTo({ top: 0, behavior: 'smooth' });
  } catch(e){
    setMsg('<div class="room-status err">Backend search unavailable (' + e.message +
           '). Falling back to the in-page agent.</div>');
    if (typeof localRunAgent === 'function') localRunAgent(q);
  }
};

/* ---- Shop a look through the backend vision pipeline ---- */
window.lookAnalyseBackend = async function(file, hostId){
  const host = document.getElementById(hostId || 'look-out');
  const say = h => { if (host) host.innerHTML = h; };
  say('<div class="adisc">Looking at the photo&hellip;</div>');
  try{
    await syncCatalogue();
    const r = await SM.look(file, shopperAudience() || '');
    if (!r.items.length){ say('<div class="adisc">No clothing was detected in that photo.</div>'); return r; }
    say('<div class="alabel" style="margin-top:0">DETECTED &middot; ' +
      (r.detector === 'vision' ? 'vision model' : 'colour and placement') + '</div>' +
      '<div class="adisc">' + r.note + '</div>' +
      r.items.map(it => {
        const d = it.detected;
        return '<div class="look-item"><div class="look-item-h"><b>' +
          (d.label || d.slot) + '</b> <span>' + d.slot +
          (d.colour ? ' &middot; ' + d.colour : '') +
          ' &middot; confidence ' + Math.round((d.confidence || 0) * 100) + '%</span></div>' +
          '<div class="look-matches">' + it.matches.map(m =>
            '<button data-detail="' + m.id + '" class="look-match">' +
            '<img src="' + (m.image || '') + '" alt="" onerror="this.style.visibility=\'hidden\'">' +
            '<span><b>' + m.name + '</b>' + (m.colour ? '<br>' + m.colour : '') +
            '<br>' + (m.why || []).join('; ') + '</span></button>').join('') + '</div></div>';
      }).join('') +
      '<div class="fit-foot">' + r.limits + '</div>');
    return r;
  } catch(e){
    say('<div class="room-status err">Shop a look needs the backend (' + e.message + ').</div>');
    throw e;
  }
};

/* ---- Model settings: provider, Gemini key and model pickers ---- */
async function settingsHTML(){
  let st = null, err = '';
  try { st = await SM.settings(); } catch(e){ err = e.message; }
  if (!st){
    return '<button class="close-tone" data-llmclose="1">&times;</button>' +
      '<div class="kicker">STYLEMATE &middot; AGENT MODEL</div><h3>Backend not reachable</h3>' +
      '<div class="room-status err">' + esc2(err) + '</div>' +
      '<div class="cfg-note">Start it with <code>uvicorn app.main:app --port 8000</code> ' +
      'in the backend folder, then reopen this panel. The page is looking at <code>' +
      esc2(SM.base) + '</code>.</div>' +
      '<label class="size-field" style="margin-top:10px">BACKEND ADDRESS' +
      '<input id="cfg-api" value="' + esc2(SM.base) + '"></label>' +
      '<button class="abtn" data-apisave="1">Save address</button>';
  }
  const g = st.gemini || {}, o = st.ollama || {};
  const lists = g.modelLists || st.geminiFallbackModels || {};
  const opts = (arr, cur) => (arr || []).map(m =>
    '<option value="' + esc2(m) + '"' + (m === cur ? ' selected' : '') + '>' + esc2(m) + '</option>').join('');
  const dot = ok => '<span class="sdot ' + (ok ? 'on' : 'off') + '"></span>';
  return '<button class="close-tone" data-llmclose="1">&times;</button>' +
    '<div class="kicker">STYLEMATE &middot; AGENT MODEL</div>' +
    '<h3 style="margin:6px 0">Which model does the thinking?</h3>' +
    '<div class="prov-row">' +
      ['auto','gemini','ollama'].map(p =>
        '<button class="room-chip ' + (st.setting === p ? 'on' : '') + '" data-prov="' + p + '">' +
        (p === 'auto' ? 'Auto' : p === 'gemini' ? 'Google Gemini' : 'Local Ollama') + '</button>').join('') +
    '</div>' +
    '<div class="adisc">Active right now: <b>' + esc2(st.active) + '</b>. ' +
      'Auto uses Gemini when a key is saved, and the local model otherwise.</div>' +

    '<div class="prov-card"><div class="prov-h">' + dot(g.reachable) + 'Google Gemini</div>' +
      '<label class="size-field">API KEY' +
        '<input id="cfg-gemkey" type="password" placeholder="' +
        (g.hasKey ? 'A key is saved - type a new one to replace it' : 'AIza...') + '"></label>' +
      '<div class="adisc">The key is sent once to your own backend, held in memory and never ' +
        'written to disk, logged, or sent back to this page.</div>' +
      '<div class="two-col">' +
        '<label class="size-field">TEXT MODEL<select id="cfg-gemtext">' + opts(lists.text, g.textModel) + '</select></label>' +
        '<label class="size-field">VISION MODEL<select id="cfg-gemvis">' + opts(lists.vision, g.visionModel) + '</select></label>' +
      '</div>' +
      (g.error ? '<div class="room-status err">' + esc2(g.error) + '</div>' : '') +
    '</div>' +

    '<div class="prov-card"><div class="prov-h">' + dot(o.reachable) + 'Local Ollama</div>' +
      '<label class="size-field">HOST<input id="cfg-ohost" value="' + esc2(st.ollama.host || '') + '"></label>' +
      '<div class="two-col">' +
        '<label class="size-field">TEXT MODEL<input id="cfg-otext" value="' + esc2(o.textModel || '') + '"></label>' +
        '<label class="size-field">VISION MODEL<input id="cfg-ovis" value="' + esc2(o.visionModel || '') + '"></label>' +
      '</div>' +
      '<div class="adisc">' + (o.reachable
        ? 'Pulled: ' + esc2((o.models || []).join(', ') || 'none')
        : 'Not reachable. Start it with <code>ollama serve</code>.') + '</div>' +
    '</div>' +
    '<button class="abtn" data-llmsave="1">Save model settings &#8599;</button>' +
    '<div class="room-status" id="cfg-status"></div>';
}

window.llmModal = async function(){
  const card = document.getElementById('llm-modal-card');
  const modal = document.getElementById('llm-modal');
  if (!card || !modal) return;
  card.innerHTML = '<div class="adisc">Loading model settings&hellip;</div>';
  modal.hidden = false;
  card.innerHTML = await settingsHTML();
};

document.addEventListener('click', async e => {
  const b = e.target.closest('button'); if (!b) return;
  const d = b.dataset || {};
  if (d.llmclose){ const m = document.getElementById('llm-modal'); if (m) m.hidden = true; return; }
  if (d.apisave){
    const v = (document.getElementById('cfg-api') || {}).value;
    if (v){ try{ localStorage.setItem('sm_api', v.trim()); }catch(err){} SM.base = v.trim(); }
    window.llmModal(); return;
  }
  if (d.prov){
    try{ await SM.saveSettings({ provider: d.prov }); }catch(err){}
    window.llmModal(); return;
  }
  if (d.llmsave){
    const val = id => (document.getElementById(id) || {}).value || '';
    const cfg = {};
    if (val('cfg-gemkey').trim()) cfg.gemini_api_key = val('cfg-gemkey').trim();
    if (val('cfg-gemtext')) cfg.gemini_text_model = val('cfg-gemtext');
    if (val('cfg-gemvis')) cfg.gemini_vision_model = val('cfg-gemvis');
    if (val('cfg-ohost')) cfg.ollama_host = val('cfg-ohost').trim();
    if (val('cfg-otext')) cfg.text_model = val('cfg-otext').trim();
    if (val('cfg-ovis')) cfg.vision_model = val('cfg-ovis').trim();
    const st = document.getElementById('cfg-status');
    if (st){ st.textContent = 'Saving...'; st.className = 'room-status busy'; }
    try{
      const r = await SM.saveSettings(cfg);
      if (st){
        st.textContent = 'Saved. Active provider: ' + r.active +
          (r.active === 'gemini'
            ? (r.gemini.reachable ? ' - key accepted.' : ' - but Gemini is not reachable: ' + (r.gemini.error || ''))
            : (r.ollama.reachable ? ' - Ollama reachable.' : ' - Ollama is not running.'));
        st.className = 'room-status ' + ((r.active === 'gemini' ? r.gemini.reachable : r.ollama.reachable) ? 'ok' : 'err');
      }
      const k = document.getElementById('cfg-gemkey'); if (k) k.value = '';
    } catch(err){
      if (st){ st.textContent = 'Could not save: ' + err.message; st.className = 'room-status err'; }
    }
    return;
  }
});

/* Sync the catalogue to the backend once it has finished loading. */
window.addEventListener('load', () => setTimeout(() => syncCatalogue().catch(()=>{}), 2500));

/* ===================================================================
   COLOUR STUDIO  ->  seasonal palette + real product recommendations
   =================================================================== */

/* Product thumbnails must go through the page's own img(), which falls back to
   a generated colour swatch when the dataset has no photo. Using the raw
   `image` field gave empty boxes. */
function thumb(id, cls){
  const p = Gall().find(x => String(x.id) === String(id));
  if (p){
    const html = Gcall('img', p, cls || '');
    if (html) return html;
  }
  return '<span class="thumb-missing" aria-hidden="true"></span>';
}
window.thumb = thumb;

const PAL = { busy:false, season:null, data:null, key:0, pinnedFor:null, colour:null };
window.PAL = PAL;

/* The season's colour chips are filters (page v15+): All shows the rows as the
   backend picked them; a colour shows every product in that part of an outfit
   that is the colour or close to it, from the whole catalogue in the page, for
   the same audience. Close = similar hue for colourful shades, similar
   lightness for neutral ones. */
// Season colours the catalogue's own table does not list. Without these they
// read as grey, and the filter would match grey products.
const SEASON_HEX = { 'aqua': '#5ccfd0', 'mint green': '#9ddfc0', 'camel': '#c19a6b' };
function colourCss(name){
  const extra = SEASON_HEX[String(name || '').toLowerCase()];
  if (extra) return extra;
  const C = G('CMAP') || {};
  const hit = C[name] || C[Object.keys(C).find(k => k.toLowerCase() === String(name || '').toLowerCase())];
  return (hit && hit.hex) || (window.COLOUR_HEX_CSS ? COLOUR_HEX_CSS(name) : '#ccc');
}
function colourRgb(name){
  const h = colourCss(name).replace('#', '');
  return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16) / 255);
}
function colourHsl(name){
  const [r, g, b] = colourRgb(name);
  const mx = Math.max(r,g,b), mn = Math.min(r,g,b), l = (mx+mn)/2, d = mx-mn;
  const s = d ? d / (1 - Math.abs(2*l - 1)) : 0;
  let hue = 0;
  if (d){ hue = mx === r ? ((g-b)/d) % 6 : mx === g ? (b-r)/d + 2 : (r-g)/d + 4; hue = (hue*60 + 360) % 360; }
  return { h: hue, s, l };
}
function colourLab(name){                              // sRGB -> CIE Lab (D65)
  const [r, g, b] = colourRgb(name).map(v => v > 0.04045 ? Math.pow((v + 0.055) / 1.055, 2.4) : v / 12.92);
  const f = t => t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;
  const x = f((r*0.4124 + g*0.3576 + b*0.1805) / 0.95047), y = f(r*0.2126 + g*0.7152 + b*0.0722),
        z = f((r*0.0193 + g*0.1192 + b*0.9505) / 1.08883);
  return [116*y - 16, 500*(x - y), 200*(y - z)];
}
/* Close shade: two vivid colours within 15 degrees of hue and similar
   lightness; otherwise (either one muted - beige, nude, grey) a small
   perceptual difference (CIE76 dE <= 10). Calibrated on the catalogue's
   colours: Peach -> Peach, Coral, Rose Gold, Nude; Turquoise -> Turquoise,
   Teal; Mustard -> Yellow, Gold, Mustard, Tan. */
function colourNear(want, have){
  if (!have || /not specified|multi/i.test(have)) return false;
  if (String(want).toLowerCase() === String(have).toLowerCase()) return true;
  const a = colourHsl(want), b = colourHsl(have);
  const vivid = c => c.s > 0.35 && c.l > 0.12 && c.l < 0.88;
  if (vivid(a) && vivid(b)){
    const dh = Math.min(Math.abs(a.h - b.h), 360 - Math.abs(a.h - b.h));
    return dh <= 15 && Math.abs(a.l - b.l) <= 0.18;
  }
  const A = colourLab(want), B = colourLab(have);
  return Math.hypot(A[0]-B[0], A[1]-B[1], A[2]-B[2]) <= 10;
}
function paletteAudience(){
  const sf = (G('tone') || {}).shopFor;
  return (sf === 'women' || sf === 'men') ? sf : sf === 'any' ? null : shopperAudience();
}
function paletteRowsBody(d){
  const card = p => '<button class="pal-card" data-detail="' + esc2(p.id) + '" title="' + esc2(p.why) + '">' +
    thumb(p.id, 'pal-thumb') +
    '<span><b>' + esc2(p.name) + '</b><em>' + esc2(p.colour) + ' &middot; Rs ' +
    Math.round(p.price||0) + '</em></span></button>';
  if (!PAL.colour){
    return (d.rows || []).map(row =>
      '<div class="pal-row"><div class="alabel">' + esc2(row.label.toUpperCase()) +
      ' &middot; ' + row.total + ' IN YOUR COLOURS</div><div class="pal-strip">' +
      row.items.map(card).join('') + '</div></div>').join('');
  }
  const want = PAL.colour, aud = paletteAudience();
  const rows = (d.rows || []).map(row => {
    const items = Gall()
      .filter(p => p.slot === row.slot && colourNear(want, p.colour) && (!aud || p.audience === aud))
      .sort((a, b) => ((b.colour === want) - (a.colour === want)) || ((b.score || 0) - (a.score || 0)))
      .map(p => Object.assign({}, p, { why: p.colour === want ? want + ', from your palette'
                                                             : p.colour + ' is close to ' + want }));
    return { row, items };
  }).filter(x => x.items.length);
  if (!rows.length)
    return '<div class="pal-empty">Nothing in ' + esc2(want) + ' or a close shade here yet. Try another colour, or All.</div>';
  return rows.map(({ row, items }) =>
    '<div class="pal-row"><div class="alabel">' + esc2(row.label.toUpperCase()) + ' &middot; ' + items.length +
    ' IN ' + esc2(want.toUpperCase()) + ' &amp; CLOSE SHADES</div><div class="pal-strip">' +
    items.map(card).join('') + '</div></div>').join('');
}

function paletteRowsHTML(d){
  const s = d.season;
  const chip = c => '<button type="button" class="pal-chip' + (PAL.colour === c ? ' on' : '') +
    '" data-palcol="' + esc2(c) + '"><i style="background:' + colourCss(c) + '"></i>' + esc2(c) + '</button>';
  return '<div class="pal-head">' +
      '<div class="alabel" style="margin-top:0">YOUR SEASON</div>' +
      '<div class="pal-season">' + esc2(s.name) + '</div>' +
      '<div class="pal-sum">' + esc2(s.summary) + '</div>' +
      '<div class="pal-chips"><button type="button" class="pal-chip' + (PAL.colour ? '' : ' on') +
        '" data-palcol="">All</button>' + s.colours.slice(0,12).map(chip).join('') + '</div>' +
      '<div class="pal-meta"><b>Neutrals</b> ' + s.neutrals.join(', ') +
        ' &nbsp;·&nbsp; <b>Metals</b> ' + s.metals.join(', ') +
        ' &nbsp;·&nbsp; <b>Go easy on</b> ' + s.avoid.join(', ') + '</div>' +
      (d.alternatives && d.alternatives.length
        ? '<div class="pal-alt">Not quite right? Try ' + d.alternatives.map(a =>
            '<button class="pal-altbtn" data-palseason="' + esc2(a.key) + '">' +
            esc2(a.name) + '</button>').join(' or ') + '.</div>' : '') +
    '</div>' +
    '<div id="pal-rows">' + paletteRowsBody(d) + '</div>' +
    '<div class="pal-foot">' + esc2(d.note) + '</div>' +
    '<button class="abtn" data-palapply="1" style="margin-top:10px">' +
      'Show only these colours across the whole catalogue &#8599;</button>';
}

async function refreshPalette(seasonKey){
  const host = document.getElementById('tone-preview');
  if (!host) return;
  const t = G('tone') || {};
  const undertone = t.undertone, depth = t.depth;
  if (!undertone){
    host.innerHTML = '<div class="adisc">Pick an undertone, or use a photo, ' +
      'and StyleMate will work out your season and show you what to buy in it.</div>';
    return;
  }
  // If the reading changed, a season the shopper pinned earlier no longer
  // applies - otherwise picking "Warm Autumn" would freeze the panel and
  // changing the undertone would appear to do nothing.
  const reading = undertone + '|' + (depth || 'medium');
  if (seasonKey){ PAL.pinnedFor = reading; }
  else if (PAL.pinnedFor && PAL.pinnedFor !== reading){
    PAL.season = null; PAL.pinnedFor = null;
  }
  const effective = (PAL.pinnedFor === reading) ? (seasonKey || PAL.season) : null;

  const my = ++PAL.key;
  host.innerHTML = '<div class="adisc">Working out your season and pulling ' +
    'matching products&hellip;</div>';
  try{
    await syncCatalogue();
    // The studio's own "Shopping for" choice wins while it is open, so picking
    // Women's re-fills these rows at once, before anything is saved.
    const sf = t.shopFor;
    const d = await SM.json('/api/palette', {
      undertone, depth: depth || 'medium', season: effective || null,
      audience: (sf === 'women' || sf === 'men') ? sf : sf === 'any' ? null : shopperAudience(),
      perRow: 8
    });
    if (my !== PAL.key) return;          // a newer request won
    PAL.data = d; PAL.season = d.season.key; PAL.colour = null;
    host.innerHTML = paletteRowsHTML(d);
  } catch(e){
    if (my !== PAL.key) return;
    host.innerHTML = '<div class="room-status err">Could not reach the backend (' +
      esc2(e.message) + '), so only the swatches are available.</div>' +
      '<div class="adisc">Start it with Start_StyleMate.bat and reopen this panel.</div>';
  }
}
window.refreshPalette = refreshPalette;

/* The page calls tonePreview() whenever the undertone or depth changes, from
   the swatch buttons and from the photo reading. Overriding it is the single
   hook that covers both. */
window.tonePreview = function(){
  // No argument: only an explicit click on an alternative season pins one.
  // Passing PAL.season back in here made every refresh re-pin, so changing
  // the undertone afterwards appeared to do nothing.
  refreshPalette();
};

document.addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  const d = b.dataset || {};
  if (d.palseason){ PAL.season = d.palseason; refreshPalette(d.palseason); return; }
  if ('palcol' in d && PAL.data){
    PAL.colour = d.palcol || null;
    document.querySelectorAll('[data-palcol]').forEach(x =>
      x.classList.toggle('on', (x.dataset.palcol || null) === PAL.colour));
    const rows = document.getElementById('pal-rows');
    if (rows) rows.innerHTML = paletteRowsBody(PAL.data);
    return;
  }
  if (d.palapply){
    const cols = (PAL.data && PAL.data.season)
      ? PAL.data.season.colours.concat(PAL.data.season.neutrals) : [];
    (0, eval)('assist.colours = ' + JSON.stringify(cols));
    const t = G('tone') || {};
    (0, eval)('assist.palette = ' + JSON.stringify(
      t.undertone === 'neutral' ? 'balanced' : (t.undertone || 'balanced')));
    (0, eval)('assist.depth = ' + JSON.stringify(t.depth || null));
    if (t.shopFor) Gcall('setShopFor', t.shopFor);
    Gcall('saveStyle'); Gcall('closeToneStudio'); Gcall('renderAssist');
    applyPaletteToResults(cols);
    return;
  }
});

/* Re-rank whatever the shopper is currently looking at so their colours come
   first. If they ran a search, that search's results are re-ordered; if they
   are browsing a category, the grid is re-ordered. The set never shrinks to
   nothing - items outside the palette drop to the back rather than vanish. */
function applyPaletteToResults(cols){
  const lower = cols.map(c => String(c).toLowerCase());
  const rank = {};
  lower.forEach((c, i) => { rank[c] = i; });
  const had = (0, eval)('typeof agentResults !== "undefined" && agentResults ? agentResults.length : 0');
  if (had){
    (0, eval)('agentResults = agentResults.slice().sort(function(a,b){' +
      'var R=' + JSON.stringify(rank) + ';' +
      'var ra=R[(a.colour||"").toLowerCase()]; var rb=R[(b.colour||"").toLowerCase()];' +
      'ra = (ra===undefined) ? 999 : ra; rb = (rb===undefined) ? 999 : rb;' +
      'return ra-rb;})');
    Gcall('render');
    const season = PAL.data ? PAL.data.season.name : 'your palette';
    Gcall('toast', 'Your ' + had + ' results re-ordered for ' + season);
    const resp = document.getElementById('agent-response');
    if (resp && PAL.data){
      const note = document.createElement('div');
      note.className = 'pal-applied';
      note.innerHTML = '<b>' + esc2(PAL.data.season.name) + ' applied.</b> ' +
        'These results are now ordered by what suits your colouring &mdash; ' +
        esc2(PAL.data.season.colours.slice(0,5).join(', ')) + ' first.';
      const old = resp.querySelector('.pal-applied');
      if (old) old.remove();
      resp.appendChild(note);
    }
  } else {
    Gcall('render');
    Gcall('toast', (PAL.data ? PAL.data.season.name : 'Palette') +
      ' applied &mdash; search for something and your colours come first');
  }
}
window.applyPaletteToResults = applyPaletteToResults;

/* Undertone/depth buttons re-select, then the page calls tonePreview, which is
   ours now. Nothing else to wire. */

/* ===================================================================
   SHOP A LOOK  ->  backend recognition
   The photo goes to the backend ONCE, shrunk to 768px: one vision call turns
   it into a text description of every piece. Matching against the 1,000
   SKUs then runs in Python on that text (embeddings + keywords + type and
   colour). Switching between Women, Men and All re-matches from the
   text alone - no image is sent again.
   =================================================================== */
const LOOK_MAX_EDGE = 768;
function canvasToBlob(c, maxEdge){
  return new Promise(res => {
    let src = c;
    const scale = Math.min(1, (maxEdge || LOOK_MAX_EDGE) / Math.max(c.width, c.height));
    if (scale < 1){
      src = document.createElement('canvas');
      src.width = Math.round(c.width * scale); src.height = Math.round(c.height * scale);
      src.getContext('2d').drawImage(c, 0, 0, src.width, src.height);
    }
    if (src.toBlob) src.toBlob(b => res(b), 'image/jpeg', 0.85);
    else res(null);
  });
}

const LOOK = { result: null, mode: 'all', first: 'all', from: 'none', byMode: {} };
const pageLookAnalyse = window.lookAnalyse;      /* colour-only fallback */

function lookUsageLine(r, rematch){
  const u = r.usage || {}, v = u.vision || {}, s = u.semantic || {};
  const n = (r.items || []).length;
  const parts = [];
  if (rematch) parts.push('Re-matched from the text description · no image sent');
  else if (v.cached) parts.push('This photo was already read · no image sent this time');
  else if (v.called) parts.push('1 image call to ' + esc2(v.model || 'the vision model') +
      ' · ' + (v.promptTokens || 0).toLocaleString('en-IN') + ' in / ' +
      (v.outputTokens || 0).toLocaleString('en-IN') + ' out tokens' +
      (v.thoughtTokens ? ' (+' + v.thoughtTokens + ' thinking)' : '') +
      ' · ' + (v.seconds || 0).toFixed(1) + 's' +
      (v.imagePx && v.imagePx[0] ? ' · photo sent at ' + v.imagePx.join('×') : ''));
  parts.push(n + ' piece' + (n === 1 ? '' : 's') + ' matched against ' +
    (u.catalogueSkus || '') + ' SKUs ' + (s.used
      ? 'by meaning (' + esc2(s.model || 'embeddings') + ', ' + (s.seconds || 0).toFixed(1) + 's) and keywords'
      : 'by keywords, type and colour' + (s.error ? ' (semantic search unavailable)' : '')));
  return parts.join(' · ');
}

function lookMatchHTML(m){
  const fmt = G('fmt');
  const price = typeof fmt === 'function' ? fmt(m.price || 0) : 'Rs ' + Math.round(m.price || 0);
  const rating = m.rating ? ' · ★ ' + m.rating : '';
  return '<button class="look-match" data-detail="' + esc2(m.id) + '" title="' + esc2(m.name) + '">' +
    thumb(m.id, 'look-thumb') +
    '<span><b>' + esc2(m.name) + '</b><br>' + esc2(m.brand || '') +
    '<br><b style="color:#2b2a26">' + price + '</b>' + rating +
    '<br><em style="font-style:normal;color:#55634a">' + (m.score || 0) + '% match</em> · ' +
    esc2((m.why || []).join('; ')) + '</span></button>';
}

function renderLook(r, rematch){
  const out = document.getElementById('look-out');
  if (!out) return;
  const mode = LOOK.mode;
  const chip = (k, label) => '<button class="room-chip' + (mode === k ? ' on' : '') +
    '" data-lookfor="' + k + '" type="button">' + label + '</button>';
  const items = r.items || [];
  out.innerHTML =
    '<div class="alabel" style="margin-top:0">' + (r.detector === 'vision' ? 'WHAT THE AI SEES' : 'COLOUR AND PLACEMENT') + '</div>' +
    (r.summary ? '<div class="look-summary">' + esc2(r.summary) + '</div>' : '') +
    '<div class="look-for">Show ' + chip('women', 'Women') + chip('men', 'Men') + chip('all', 'All') +
      (mode === LOOK.first && LOOK.from !== 'none'
        ? '<span style="font-size:9px;color:#767970;margin-left:6px">' +
          (LOOK.from === 'photo' ? 'chosen from the photo' : 'your saved choice') + '</span>' : '') +
      '</div>' +
    '<div class="look-usage">' + lookUsageLine(r, rematch) + '</div>' +
    items.map(it => {
      const d = it.detected || {};
      const det = (d.details || []).slice(0, 4);
      return '<div class="look-item"><div class="look-item-h"><b>' +
        esc2(d.description || d.label || d.slot) + '</b><span>' + esc2(d.slot) +
        (d.article_type ? ' · ' + esc2(d.article_type) : '') +
        (d.colour ? ' · ' + esc2(d.colour) : '') +
        (d.pattern && d.pattern !== 'solid' ? ' · ' + esc2(d.pattern) : '') +
        ' · ' + Math.round((d.confidence || 0) * 100) + '% sure</span>' +
        (det.length ? '<div class="look-details">' + det.map(x => '<i>' + esc2(x) + '</i>').join('') + '</div>' : '') +
        '</div><div class="look-matches">' +
        ((it.matches || []).length ? it.matches.map(lookMatchHTML).join('')
          : '<span class="adisc">' + (mode === 'women' || mode === 'men'
              ? 'No ' + (mode === 'women' ? 'women’s' : 'men’s') + ' product like this in the catalogue. Try All.'
              : 'Nothing like this on the shelf.') + '</span>') +
        '</div></div>';
    }).join('') +
    '<div class="fit-foot">' + esc2(r.limits || '') + '</div>';
  /* "Add the look to cart" takes the best match for each piece. */
  const look = G('look'), all = Gall();
  if (look){
    look.items = items.map(it => (it.matches || [])[0]).filter(Boolean)
      .map(m => all.find(p => p.id === m.id)).filter(Boolean);
  }
}

function lookAudit(label, r, ms, err, px){
  const open = G('logOpen'), close = G('logClose');
  if (typeof open !== 'function' || typeof close !== 'function') return;
  try{
    const rec = open({ model: '(backend decides)', transport: 'backend',
      messages: [{ role: 'user', content: '[outfit photo ' + (px || '') + ' - sent once to the vision model; the image itself is not logged]' }] },
      label, true);
    rec.endpoint = label === 'shop-a-look' ? '/api/look' : '/api/look/match';
    if (err){ close(rec, { status: 'error', error: err, error_kind: 'backend', latency_ms: ms }); return; }
    const u = r.usage || {}, v = u.vision || {}, s = u.semantic || {};
    close(rec, { status: 'ok', http_status: 200, latency_ms: ms,
      prompt_tokens: v.cached || !v.called ? 0 : (v.promptTokens || 0),
      completion_tokens: v.cached || !v.called ? 0 : (v.outputTokens || 0),
      served_by_model: v.model || null, provider: 'backend',
      response_text: JSON.stringify({ summary: r.summary, items: (r.items || []).map(i => i.detected) }),
      vision_cached: !!v.cached, semantic_used: !!s.used, semantic_model: s.model || null,
      products_shown: (r.items || []).reduce((n, i) => n + (i.matches || []).length, 0),
      shown_ids: (r.items || []).flatMap(i => (i.matches || []).map(m => m.id)).slice(0, 60) });
  } catch(e){ /* auditing must never break the feature */ }
}

window.lookAnalyse = async function(){
  const c = document.getElementById('look-canvas');
  const out = document.getElementById('look-out');
  const stat = document.getElementById('look-status');
  if (!c || c.hidden || !out) return;
  const say = h => { out.innerHTML = h; };
  if (stat) stat.textContent = 'Reading the photo: one AI call, then matching runs on its text description.';
  say('<div class="adisc">Looking at the photo&hellip; the AI is naming each piece.</div>');
  const t0 = performance.now();
  let px = '';
  try{
    const blob = await canvasToBlob(c, LOOK_MAX_EDGE);
    if (!blob) throw new Error('Could not read the image from the canvas.');
    const sc = Math.min(1, LOOK_MAX_EDGE / Math.max(c.width, c.height));
    px = Math.round(c.width * sc) + 'x' + Math.round(c.height * sc);
    const file = new File([blob], 'look.jpg', { type: 'image/jpeg' });
    // The backend opens on the person in the photo - a girl's photo shows
    // women's products only, a man's men's only. The saved choice (Colour
    // Studio) is sent as the fallback for a photo with no clear person.
    const r = await SM.look(file, shopperAudience() || '');
    LOOK.result = r; LOOK.first = LOOK.mode = r.audience || 'all';
    LOOK.from = r.audienceFrom || 'none';
    LOOK.byMode = { [LOOK.mode]: r };
    lookAudit('shop-a-look', r, Math.round(performance.now() - t0), null, px);
    if (stat) stat.textContent = r.detector === 'vision'
      ? (r.usage && r.usage.vision && r.usage.vision.cached
          ? 'Already read this photo, so it was not sent to the AI again.'
          : 'Read once by the AI. Everything below was matched from its text description.')
      : 'Read by colour and placement - the vision model was not available.';
    if (!(r.items || []).length){
      say('<div class="adisc">' + esc2(r.note || 'No clothing was detected in that photo.') +
          ' A clear, well-lit, full-outfit shot works best.</div>');
      return;
    }
    renderLook(r, false);
    Gcall('toast', 'Found ' + r.items.length + ' pieces · matched against ' + ((r.usage || {}).catalogueSkus || '') + ' products');
  } catch(e){
    lookAudit('shop-a-look', null, Math.round(performance.now() - t0), e.message, px);
    if (stat) stat.textContent = 'AI recognition unavailable - matched by colour instead.';
    if (typeof pageLookAnalyse === 'function'){
      try { pageLookAnalyse(); } catch(e2){}
      out.insertAdjacentHTML('afterbegin', '<div class="room-status err">The StyleMate backend did not answer (' +
        esc2(e.message) + '), so these are colour matches only. Start it with: bash start_stylemate.command</div>');
    } else {
      say('<div class="room-status err">Shop a Look needs the StyleMate backend (' + esc2(e.message) + ').</div>');
    }
  }
};

/* Women / Men / All: re-run the match from the text description. */
document.addEventListener('click', async e => {
  const b = e.target.closest && e.target.closest('[data-lookfor]');
  if (!b || !LOOK.result) return;
  const mode = b.dataset.lookfor;
  if (mode === LOOK.mode) return;
  LOOK.mode = mode;
  const t0 = performance.now();
  if (LOOK.byMode[mode]){ renderLook(LOOK.byMode[mode], mode !== LOOK.first); return; }
  const u = document.querySelector('.look-usage');
  if (u) u.textContent = 'Re-matching ' + (mode === 'all' ? 'across all products' : 'for ' + mode + ' only') + ' from the text description...';
  try{
    // women / men: that section only. all: no filter.
    const r = await SM.json('/api/look/match', {
      items: LOOK.result.items.map(i => i.detected), audience: mode, perItem: 6 });
    const merged = Object.assign({}, LOOK.result, { items: r.items, usage: r.usage });
    LOOK.byMode[mode] = merged;
    lookAudit('shop-a-look-rematch', merged, Math.round(performance.now() - t0), null, '');
    renderLook(merged, true);
  } catch(err){
    if (u) u.textContent = 'Could not re-match (' + err.message + ').';
  }
});

/* ---- keep the fitting room in step with the catalogue ---- */
const _addToFitQueue = G('addToFitQueue');
window.addToFitQueue = function(id){
  if (typeof _addToFitQueue === 'function') _addToFitQueue(id);
  if (window.roomRefreshQueue) setTimeout(() => window.roomRefreshQueue(), 60);
};

/* ---- colour swatch -> CSS, for the palette chips ---- */
window.COLOUR_HEX_CSS = function(name){
  const h = hexFor(name);
  return '#' + h.toString(16).padStart(6, '0');
};

/* ===================================================================
   MINI MANNEQUIN  ->  the Fitting room card in the left sidebar (page v13+)
   A small, slowly turning copy of the fitting-room avatar: the same mesh from
   /api/avatar (parametric, no AI call, ~15 ms), drawn by its own tiny
   renderer - no floor, shadows or controls. It renders only while the card is
   on screen and the tab is visible, and holds still for prefers-reduced-
   motion. When the fitting room builds an avatar, the mini takes that mesh, so
   it always matches. Without WebGL or a backend the card keeps its drawing.
   =================================================================== */
const MINI = { host: null, renderer: null, scene: null, camera: null, body: null,
               raf: 0, t: 0, onScreen: true, still: false, dirty: true };
window.MINI = MINI;

function miniInit(){
  const host = document.getElementById('mini-mannequin');
  if (!host || MINI.renderer) return;
  const canvas = document.createElement('canvas');
  try {
    MINI.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
  } catch (e) { return; }                          // no WebGL: the drawing stays
  MINI.host = host;
  host.appendChild(canvas);
  MINI.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  const w = host.clientWidth || 30, h = host.clientHeight || 66;
  MINI.renderer.setSize(w, h, false);
  MINI.scene = new THREE.Scene();
  MINI.scene.add(new THREE.HemisphereLight(0xffffff, 0xd8d2c4, 0.85));
  const key = new THREE.DirectionalLight(0xfff6e8, 1.45); key.position.set(2.4, 3.4, 2.8);
  const rim = new THREE.DirectionalLight(0xffffff, 0.7); rim.position.set(-0.6, 2.4, -3.2);
  MINI.scene.add(key, rim);
  MINI.camera = new THREE.PerspectiveCamera(30, w / h, 0.05, 30);
  const rm = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)');
  MINI.still = !!(rm && rm.matches);
  if (rm && rm.addEventListener) rm.addEventListener('change', e => { MINI.still = e.matches; });
  if ('IntersectionObserver' in window){
    new IntersectionObserver(es => { MINI.onScreen = es[0].isIntersecting; }).observe(host);
  }
  SM.json('/api/avatar', { profile: bodyProfile() })
    .then(d => miniSetBody(d && d.mesh))
    .catch(() => {});                              // no backend: the drawing stays
}

function miniSetBody(mesh){
  if (!MINI.scene || !mesh || !mesh.positions) return;
  const turn = MINI.body ? MINI.body.rotation.y : 0.5;
  if (MINI.body){ MINI.scene.remove(MINI.body); MINI.body.geometry.dispose(); MINI.body.material.dispose(); }
  MINI.body = new THREE.Mesh(Viewer.geom(mesh.positions, mesh.indices),
    new THREE.MeshStandardMaterial({ color: 0xd8b79a, roughness: 0.78, metalness: 0.02 }));
  MINI.body.rotation.y = turn;
  MINI.scene.add(MINI.body);
  // Whole figure in frame, with room to turn: the arms never leave the canvas.
  const H = mesh.heightM || 1.7;
  MINI.camera.position.set(0, H * 0.56, H * 2.1);
  MINI.camera.lookAt(0, H * 0.5, 0);
  MINI.dirty = true;
  MINI.host.classList.add('live');
  if (!MINI.raf) miniLoop();
}

function miniLoop(now){
  MINI.raf = requestAnimationFrame(miniLoop);
  if (!MINI.onScreen || document.hidden || !MINI.body){ MINI.t = 0; return; }
  // Time-based, so it turns at the same speed on a 60 Hz or 120 Hz screen,
  // and does not jump after a pause: about 9 seconds a turn.
  const dt = MINI.t && now ? Math.min(0.1, (now - MINI.t) / 1000) : 0;
  MINI.t = now || 0;
  if (!MINI.still && dt){ MINI.body.rotation.y += dt * 0.7; MINI.dirty = true; }
  if (!MINI.dirty) return;
  MINI.renderer.render(MINI.scene, MINI.camera);
  MINI.dirty = false;
}
window.addEventListener('load', () => setTimeout(miniInit, 600));

/* ===================================================================
   AGENT MODEL STATUS  ->  live light in the left rail
   =================================================================== */
const STATUS = { timer: null, last: null };

function statusLineHTML(h){
  if (!h) return '';
  const g = h.gemini || {}, o = h.ollama || {};
  const active = h.active;
  const usingGemini = active === 'gemini';
  const live = usingGemini ? g.reachable : o.reachable;
  const model = usingGemini ? g.textModel : o.textModel;

  let light, headline, detail;
  if (live){
    light = 'ok';
    headline = usingGemini ? 'Gemini connected' : 'Local model connected';
    detail = usingGemini
      ? 'API key accepted. Search, Shop a Look and the Colour Studio are '
        + 'running on ' + esc2(model) + '.'
      : 'Ollama is answering on ' + esc2(o.host || '') + ' with '
        + esc2(model) + '.';
  } else if (usingGemini && g.hasKey){
    light = 'bad';
    headline = 'API key not working';
    detail = esc2(g.error || 'Google rejected the key or could not be reached.')
      + ' Open Agent model and check the key.';
  } else if (usingGemini){
    light = 'bad';
    headline = 'No API key saved';
    detail = 'Paste a Gemini key under Agent model. Until then the agent '
      + 'answers from rules only.';
  } else {
    light = 'warn';
    headline = 'No model connected';
    detail = 'Search still works from rules, but it cannot read a sentence '
      + 'properly or recognise clothes in a photo. Add a Gemini key under '
      + 'Agent model.';
  }

  const row = (okFlag, label, note) =>
    '<div class="st-row"><span class="st-dot ' + (okFlag ? 'ok' : 'bad') + '"></span>' +
    '<span><b>' + label + '</b>' + (note ? '<em>' + note + '</em>' : '') + '</span></div>';

  const visionReady = usingGemini ? g.visionModelReady : o.visionModelReady;
  return '<div class="st-head"><span class="st-light ' + light + '"></span>' +
      '<span class="st-headline">' + headline + '</span></div>' +
      '<div class="st-detail">' + detail + '</div>' +
      '<div class="st-rows">' +
        row(live, 'Smart search', live ? 'reads your sentence' : 'rule parsing only') +
        row(!!(live && visionReady), 'Photo recognition',
            (live && visionReady) ? esc2(usingGemini ? g.visionModel : o.visionModel)
                                  : 'colour and placement only') +
        row(true, 'Fit engine', 'always on, no model needed') +
      '</div>' +
      '<button class="st-btn" data-llmcfg="1">' +
        (live ? 'Change model' : 'Configure API / model') + ' &#8599;</button>';
}

/* The header light (page v10+): green when Gemini is the active provider and
   its key works; red when it does not, or when the backend is not running. */
function paintModelLight(h){
  const el = document.getElementById('model-light');
  if (!el) return;
  const g = (h && h.gemini) || {};
  const ok = !!(h && h.active === 'gemini' && g.reachable);
  el.classList.toggle('ok', ok);
  el.classList.toggle('bad', !ok);
  el.title = !h ? 'Not connected: the StyleMate backend is not running'
    : ok ? 'Gemini connected · ' + (g.textModel || '') +
           (g.visionModel ? ' · photos: ' + g.visionModel : '') + '. Click for model settings.'
    : g.hasKey ? 'Gemini not connected: ' + (g.error || 'the API key was rejected or Google could not be reached') +
                 '. Click to check the key.'
    : 'Gemini not connected: no API key saved. Click to add one.';
  el.setAttribute('aria-label', ok ? 'Gemini connected' : 'Gemini not connected');
}

async function refreshAgentStatus(){
  const host = document.getElementById('sm-status');
  try{
    const h = await SM.settings();
    STATUS.last = h;
    paintModelLight(h);
    if (!host) return;
    host.innerHTML = statusLineHTML(h);
    host.dataset.ok = (h.active === 'gemini' ? (h.gemini || {}).reachable
                                             : (h.ollama || {}).reachable) ? '1' : '0';
  } catch(e){
    STATUS.last = null;
    paintModelLight(null);
    if (!host) return;
    host.dataset.ok = '0';
    host.innerHTML =
      '<div class="st-head"><span class="st-light bad"></span>' +
      '<span class="st-headline">Backend not running</span></div>' +
      '<div class="st-detail">StyleMate cannot reach its Python backend at ' +
      esc2(SM.base) + '. Start it with <b>Start_StyleMate.bat</b>. ' +
      'Until then search, Shop a Look and the fitting room are unavailable.</div>' +
      '<button class="st-btn" data-llmcfg="1">Backend settings &#8599;</button>';
  }
}
window.refreshAgentStatus = refreshAgentStatus;

/* Replace the page's own static card with the live one, once. From page v10
   there is no card, only the header light; the polling runs either way. */
function mountStatusCard(){
  if (STATUS.timer) return;
  const solo = document.querySelector('.agent-solo');
  if (solo && !document.getElementById('sm-status')){
    const box = document.createElement('div');
    box.id = 'sm-status';
    box.className = 'sm-status';
    solo.replaceWith(box);
  }
  refreshAgentStatus();
  if (STATUS.timer) clearInterval(STATUS.timer);
  STATUS.timer = setInterval(refreshAgentStatus, 20000);
}
window.addEventListener('load', () => setTimeout(mountStatusCard, 900));
document.addEventListener('click', e => {
  const b = e.target.closest('button');
  if (b && (b.dataset.llmsave || b.dataset.prov)) setTimeout(refreshAgentStatus, 900);
});
