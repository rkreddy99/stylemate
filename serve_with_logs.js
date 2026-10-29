#!/usr/bin/env node
/*
 * serve_with_logs.js — Node version of the StyleMate log sink.
 *
 * Identical behaviour to serve_with_logs.py: serves this folder over HTTP and
 * writes one JSON file per Ollama call into ./logs/.
 *
 * It exists because macOS ships a /usr/bin/python3 that is only a stub: running
 * it pops the "install command line developer tools" dialog and exits, which is
 * exactly what happened here — sink.log contained an xcode-select prompt and no
 * logs were ever written. Node is usually already installed; if it is not,
 * `xcode-select --install` fixes the Python one instead. runall.sh picks
 * whichever runtime actually works.
 *
 *     node serve_with_logs.js
 *
 * Listens on 127.0.0.1 only. Nothing leaves the machine.
 */
'use strict';

const http = require('http');
const fs   = require('fs');
const path = require('path');

const PORT    = parseInt(process.env.PORT || '8765', 10);
const ROOT    = __dirname;
const LOGDIR  = path.join(ROOT, 'logs');
const MAXBODY = 64 * 1024 * 1024;   // a full record is ~90 KB; this is slack
let written   = 0;

const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
  '.png': 'image/png', '.webp': 'image/webp', '.svg': 'image/svg+xml',
  '.gif': 'image/gif', '.ico': 'image/x-icon', '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/plain; charset=utf-8',
};

/* The filename arrives from the page, so it is untrusted: keep the basename
   only, and only characters we chose to allow. Without this, a record named
   ../../etc/something would escape the logs folder. */
function safeName(name) {
  let base = path.basename(String(name || ''));
  base = base.replace(/[^A-Za-z0-9._-]/g, '');
  if (!base || base.startsWith('.')) {
    base = 'ollama-log_' + new Date().toISOString().replace(/[:.]/g, '-') + '.json';
  }
  if (!base.endsWith('.json')) base += '.json';
  return base.slice(0, 200);
}

function cors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
}

function sendJson(res, code, payload) {
  const body = Buffer.from(JSON.stringify(payload));
  res.writeHead(code, { 'Content-Type': 'application/json', 'Content-Length': body.length });
  res.end(body);
}

function serveFile(req, res) {
  let rel = decodeURIComponent((req.url || '/').split('?')[0]);
  if (rel === '/' || rel === '') rel = '/index.html';
  const full = path.join(ROOT, path.normalize(rel).replace(/^(\.\.[\\/])+/, ''));
  if (!full.startsWith(ROOT)) { sendJson(res, 403, { error: 'outside the served folder' }); return; }
  fs.readFile(full, (err, data) => {
    if (err) {
      if (rel === '/index.html') {   // no index: list the pages instead
        const pages = fs.readdirSync(ROOT).filter(f => f.endsWith('.html')).sort();
        const html = '<meta charset="utf-8"><h1>StyleMate</h1><ul>' +
          pages.map(p => '<li><a href="/' + encodeURIComponent(p) + '">' + p + '</a></li>').join('') + '</ul>';
        res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
        res.end(html);
        return;
      }
      sendJson(res, 404, { error: 'not found' });
      return;
    }
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(full).toLowerCase()] || 'application/octet-stream' });
    res.end(data);
  });
}

const server = http.createServer((req, res) => {
  cors(res);

  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  const url = (req.url || '/').split('?')[0].replace(/\/$/, '');

  if (req.method === 'GET' && url === '/ping') {
    sendJson(res, 200, { ok: true, logs: LOGDIR, written, runtime: 'node ' + process.version });
    return;
  }

  if (req.method === 'POST') {
    if (url !== '/log') { sendJson(res, 404, { error: 'post audit records to /log' }); return; }
    let size = 0;
    const chunks = [];
    req.on('data', c => {
      size += c.length;
      if (size > MAXBODY) { req.destroy(); return; }
      chunks.push(c);
    });
    req.on('end', () => {
      let payload;
      try { payload = JSON.parse(Buffer.concat(chunks).toString('utf8')); }
      catch (e) { sendJson(res, 400, { error: 'unparseable record: ' + e.message }); return; }

      const record  = payload.record || payload;
      const name    = safeName(payload.filename);
      const rewrite = !!payload.rewrite;
      const file    = path.join(LOGDIR, name);
      const existed = fs.existsSync(file);
      try {
        fs.mkdirSync(LOGDIR, { recursive: true });
        // Written whole, so a half-flushed file is never left behind.
        const tmp = file + '.part';
        fs.writeFileSync(tmp, JSON.stringify(record, null, 2), 'utf8');
        fs.renameSync(tmp, file);
      } catch (e) {
        sendJson(res, 500, { error: 'could not write ' + file + ': ' + e.message });
        return;
      }
      if (!existed) written++;
      const verb  = (existed && rewrite) ? 'updated' : 'wrote';
      const ptok  = record.prompt_tokens ? ' ' + record.prompt_tokens + ' prompt tok' : '';
      const trunc = record.prompt_truncated ? ' TRUNCATED' : '';
      console.log('  ' + verb + ' logs/' + name + '  [' + (record.status || '?') + ptok + ']' + trunc);
      sendJson(res, 200, { ok: true, path: 'logs/' + name });
    });
    return;
  }

  if (req.method === 'GET') { serveFile(req, res); return; }
  sendJson(res, 405, { error: 'method not allowed' });
});

server.on('error', err => {
  if (err.code === 'EADDRINUSE') {
    console.error('Could not bind port ' + PORT + ': already in use.');
    console.error('Try:  PORT=8766 node serve_with_logs.js');
    console.error('If you change the port, update logSink in the page to match.');
  } else {
    console.error('Server error: ' + err.message);
  }
  process.exit(1);
});

fs.mkdirSync(LOGDIR, { recursive: true });
server.listen(PORT, '127.0.0.1', () => {
  const pages = fs.readdirSync(ROOT).filter(f => f.endsWith('.html')).sort();
  console.log('StyleMate log sink (node ' + process.version + ')');
  console.log('  serving   ' + ROOT);
  console.log('  logs      ' + LOGDIR + '/');
  console.log('  listening http://localhost:' + PORT + '/  (127.0.0.1 only)');
  pages.forEach(p => console.log('    http://localhost:' + PORT + '/' + p));
  console.log('\nOpen a page above, search, and one JSON file per Ollama call appears in logs/.');
  console.log('Ctrl-C to stop.\n');
});

process.on('SIGINT', () => {
  console.log('\nStopped. ' + written + ' log file(s) written this run.');
  process.exit(0);
});
