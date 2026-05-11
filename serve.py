#!/usr/bin/env python3
"""
StudyNotes — local dev server with editing UI.

Usage:
    python3 serve.py          # default port 8080
    python3 serve.py 3000     # custom port

Browse at http://127.0.0.1:8080/
Editor at http://127.0.0.1:8080/_admin/
"""

import json
import os
import re
import sys
import time
import threading
import urllib.parse
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import yaml

from build import build, CONTENT_DIR, TEMPLATE_DIR, STATIC_DIR, OUTPUT_DIR

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
WATCH_DIRS = [CONTENT_DIR, TEMPLATE_DIR, STATIC_DIR]
_rebuild_lock = threading.Lock()


def rebuild():
    with _rebuild_lock:
        build()


# ── File-system helpers ────────────────────────────────────
def list_content():
    """Return the full content tree as dicts."""
    categories = []
    if not CONTENT_DIR.exists():
        return categories
    for cat_path in sorted(CONTENT_DIR.iterdir()):
        if not cat_path.is_dir():
            continue
        meta = _load_meta(cat_path)
        cat = {
            'name': meta.get('name', cat_path.name),
            'folder': cat_path.name,
            'description': meta.get('description', ''),
            'icon': meta.get('icon', 'bi-folder'),
            'color': meta.get('color', '#2ea052'),
            'order': meta.get('order', 0),
            'subcategories': [],
        }
        for sub_path in sorted(cat_path.iterdir()):
            if not sub_path.is_dir():
                continue
            sub_meta = _load_meta(sub_path)
            sub = {
                'name': sub_meta.get('name', sub_path.name),
                'folder': sub_path.name,
                'order': sub_meta.get('order', 0),
                'articles': [],
            }
            for md in sorted(sub_path.glob('*.md')):
                sub['articles'].append({
                    'filename': md.name,
                    'title': _quick_title(md),
                    'path': f'{cat_path.name}/{sub_path.name}/{md.name}',
                })
            cat['subcategories'].append(sub)
        categories.append(cat)
    return categories


def _load_meta(path):
    f = path / '_meta.yml'
    if f.exists():
        return yaml.safe_load(f.read_text(encoding='utf-8')) or {}
    return {}


def _quick_title(md_path):
    """Read just the title from front-matter without parsing the whole file."""
    with open(md_path, 'r', encoding='utf-8') as f:
        head = f.read(4096)
    if head.startswith('---'):
        parts = head.split('---', 2)
        if len(parts) >= 3:
            try:
                fm = yaml.safe_load(parts[1])
                if fm and 'title' in fm:
                    return fm['title']
            except Exception:
                pass
    return md_path.stem


def read_article(rel_path):
    fp = CONTENT_DIR / rel_path
    if not fp.exists():
        return None
    return fp.read_text(encoding='utf-8')


def save_article(rel_path, content):
    fp = CONTENT_DIR / rel_path
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content, encoding='utf-8')


def delete_article(rel_path):
    fp = CONTENT_DIR / rel_path
    if fp.exists():
        fp.unlink()


def create_folder(rel_path, meta):
    fp = CONTENT_DIR / rel_path
    fp.mkdir(parents=True, exist_ok=True)
    meta_file = fp / '_meta.yml'
    with open(meta_file, 'w', encoding='utf-8') as f:
        yaml.dump(meta, f, allow_unicode=True, default_flow_style=False)


def delete_folder(rel_path):
    import shutil
    fp = CONTENT_DIR / rel_path
    if fp.exists() and fp.is_dir():
        shutil.rmtree(fp)


# ── HTTP Handler ───────────────────────────────────────────
class Handler(SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(OUTPUT_DIR), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == '/_admin/' or path == '/_admin':
            self._send_html(ADMIN_HTML)
        elif path == '/_admin/api/content':
            self._send_json(list_content())
        elif path == '/_admin/api/article':
            qs = urllib.parse.parse_qs(parsed.query)
            rel = qs.get('path', [''])[0]
            text = read_article(rel)
            if text is None:
                self._send_json({'error': 'not found'}, 404)
            else:
                self._send_json({'path': rel, 'content': text})
        else:
            super().do_GET()

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        path = urllib.parse.urlparse(self.path).path

        if path == '/_admin/api/upload-image':
            # Handle image upload (raw binary with filename in header)
            content_type = self.headers.get('Content-Type', '')
            filename = self.headers.get('X-Filename', '')
            if not filename:
                filename = f'img_{int(time.time())}.png'
            # Sanitize filename
            filename = re.sub(r'[\\/:*?"<>|]', '_', filename)
            img_dir = STATIC_DIR / 'images' / 'uploads'
            img_dir.mkdir(parents=True, exist_ok=True)
            dest = img_dir / filename
            # Avoid overwriting
            if dest.exists():
                stem, ext = os.path.splitext(filename)
                filename = f'{stem}_{int(time.time())}{ext}'
                dest = img_dir / filename
            data = self.rfile.read(length)
            dest.write_bytes(data)
            # Also copy to docs/static for immediate local preview
            docs_img_dir = OUTPUT_DIR / 'static' / 'images' / 'uploads'
            docs_img_dir.mkdir(parents=True, exist_ok=True)
            (docs_img_dir / filename).write_bytes(data)
            self._send_json({'ok': True, 'path': f'/static/images/uploads/{filename}'})
            return

        body = json.loads(self.rfile.read(length)) if length else {}

        if path == '/_admin/api/save':
            rel = body.get('path', '')
            content = body.get('content', '')
            if not rel:
                self._send_json({'error': 'path required'}, 400)
                return
            save_article(rel, content)
            rebuild()
            self._send_json({'ok': True})

        elif path == '/_admin/api/delete':
            rel = body.get('path', '')
            typ = body.get('type', 'article')
            if not rel:
                self._send_json({'error': 'path required'}, 400)
                return
            if typ == 'article':
                delete_article(rel)
            else:
                delete_folder(rel)
            rebuild()
            self._send_json({'ok': True})

        elif path == '/_admin/api/new-folder':
            rel = body.get('path', '')
            meta = body.get('meta', {})
            if not rel:
                self._send_json({'error': 'path required'}, 400)
                return
            create_folder(rel, meta)
            rebuild()
            self._send_json({'ok': True})

        elif path == '/_admin/api/new-article':
            cat = body.get('category', '')
            sub = body.get('subcategory', '')
            title = body.get('title', '新文章')
            if not cat or not sub:
                self._send_json({'error': 'category and subcategory required'}, 400)
                return
            fname = re.sub(r'[\\/:*?"<>|]', '_', title) + '.md'
            rel = f'{cat}/{sub}/{fname}'
            now = datetime.now().strftime('%Y-%m-%d %H:%M')
            content = f'---\ntitle: {title}\nsummary: ""\npinned: false\ncreated: "{now}"\nupdated: "{now}"\n---\n\n'
            save_article(rel, content)
            rebuild()
            self._send_json({'ok': True, 'path': rel})

        else:
            self._send_json({'error': 'not found'}, 404)

    def _send_json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        first = str(args[0]) if args else ''
        if '/_admin/' not in first:
            super().log_message(fmt, *args)


# ── Admin SPA ──────────────────────────────────────────────
ADMIN_HTML = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>StudyNotes Editor</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Noto+Sans+SC:wght@400;500;700&display=swap" rel="stylesheet">
<!-- Preview rendering -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github.min.css">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<!-- Site CSS for preview fidelity -->
<link rel="stylesheet" href="/static/css/tokens.css">
<link rel="stylesheet" href="/static/css/code.css">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { height: 100%; overflow: hidden; }
body { font-family: 'Inter', 'Noto Sans SC', system-ui, sans-serif; background: #f2f8ee; color: #2a3729; }

/* ── Top bar ── */
.topbar { background: #fff; border-bottom: 1px solid rgba(30,80,40,.1); padding: 12px 24px; display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 100; }
.topbar .logo { font-weight: 700; font-size: 1.1rem; color: #1f7f3f; display: flex; align-items: center; gap: 8px; }
.topbar .logo i { font-size: 1.3rem; }
.topbar .back { color: #536751; text-decoration: none; font-size: .85rem; margin-left: auto; }
.topbar .back:hover { color: #1f7f3f; }

/* ── Layout ── */
.wrap { display: flex; height: calc(100vh - 49px); }
.sidebar { width: 300px; min-width: 300px; background: #fff; border-right: 1px solid rgba(30,80,40,.08); overflow-y: auto; padding: 16px 0; }
.main-panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

/* ── Sidebar tree ── */
.cat-group { margin-bottom: 4px; }
.cat-header { display: flex; align-items: center; gap: 8px; padding: 8px 16px; font-weight: 600; font-size: .88rem; cursor: pointer; user-select: none; }
.cat-header:hover { background: rgba(46,160,82,.06); }
.cat-header .dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.cat-header .count { font-size: .72rem; color: #8a9b86; margin-left: auto; font-weight: 400; }
.sub-group { margin-left: 20px; }
.sub-header { display: flex; align-items: center; gap: 6px; padding: 5px 16px; font-size: .82rem; font-weight: 500; color: #536751; cursor: pointer; }
.sub-header:hover { color: #1f7f3f; }
.sub-header .count { font-size: .7rem; color: #8a9b86; margin-left: auto; }
.art-item { display: flex; align-items: center; gap: 6px; padding: 4px 16px 4px 36px; font-size: .8rem; color: #536751; cursor: pointer; text-decoration: none; border-radius: 4px; }
.art-item:hover, .art-item.active { background: rgba(46,160,82,.08); color: #1f7f3f; }
.art-item .bi-file-earmark-text { font-size: .7rem; }

/* ── Action buttons in sidebar ── */
.action-row { display: flex; gap: 4px; margin-left: auto; }
.action-row button { background: none; border: none; cursor: pointer; color: #8a9b86; font-size: .78rem; padding: 2px 4px; border-radius: 4px; }
.action-row button:hover { color: #d64560; background: rgba(214,69,96,.08); }
.action-row .add-btn:hover { color: #1f7f3f; background: rgba(46,160,82,.08); }
.sidebar-bottom { padding: 12px 16px; border-top: 1px solid rgba(30,80,40,.08); margin-top: 8px; }
.sidebar-bottom button { width: 100%; padding: 8px; border: 1px dashed rgba(30,80,40,.2); background: none; border-radius: 8px; cursor: pointer; color: #536751; font-size: .82rem; }
.sidebar-bottom button:hover { border-color: #1f7f3f; color: #1f7f3f; background: rgba(46,160,82,.04); }

/* ── Editor ── */
.editor-wrap { flex: 1; display: flex; flex-direction: column; min-height: 0; }
.editor-toolbar { padding: 12px 20px; background: #fff; border-bottom: 1px solid rgba(30,80,40,.08); display: flex; align-items: center; gap: 12px; }
.editor-toolbar h2 { font-size: 1rem; font-weight: 600; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.editor-toolbar button { padding: 6px 16px; border-radius: 8px; border: none; cursor: pointer; font-size: .82rem; font-weight: 500; }
.btn-save { background: #1f7f3f; color: #fff; }
.btn-save:hover { background: #186432; }
.btn-save:disabled { background: #8a9b86; cursor: not-allowed; }
.btn-upload { background: none; color: #1f7f3f; border: 1px solid rgba(31,127,63,.3) !important; }
.btn-upload:hover { background: rgba(31,127,63,.06); }
.btn-delete { background: none; color: #d64560; border: 1px solid rgba(214,69,96,.3) !important; }
.btn-delete:hover { background: rgba(214,69,96,.06); }
.btn-toggle { background: rgba(46,160,82,.08); color: #1f7f3f; }
.btn-toggle.active { background: #1f7f3f; color: #fff; }
.editor-body { flex: 1; display: flex; overflow: hidden; min-height: 0; }
.editor-body textarea { flex: 1; padding: 20px; border: none; resize: none; font-family: 'JetBrains Mono', monospace; font-size: .88rem; line-height: 1.7; background: #fafdf8; outline: none; min-width: 0; overflow-y: auto; }
.editor-body .preview { flex: 1; padding: 24px 28px; overflow-y: auto; border-left: 1px solid rgba(30,80,40,.08); background: #fff; min-width: 0; min-height: 0; }
/* Preview content styling */
.preview h1, .preview h2, .preview h3, .preview h4 { margin: 1.2em 0 .5em; color: #134f29; font-family: 'Inter','Noto Sans SC',sans-serif; }
.preview h1 { font-size: 1.5rem; border-bottom: 1px solid rgba(30,80,40,.1); padding-bottom: .3em; }
.preview h2 { font-size: 1.25rem; }
.preview h3 { font-size: 1.08rem; }
.preview p { margin: .6em 0; line-height: 1.8; font-size: .92rem; }
.preview ul, .preview ol { margin: .5em 0; padding-left: 1.5em; }
.preview li { margin: .25em 0; line-height: 1.7; font-size: .92rem; }
.preview code { font-family: 'JetBrains Mono', monospace; font-size: .84rem; background: #f0f6ec; padding: 2px 6px; border-radius: 4px; color: #1f7f3f; }
.preview pre { margin: 1em 0; border-radius: 10px; overflow: hidden; }
.preview pre code { display: block; padding: 16px 20px; background: #f8faf6; border: 1px solid rgba(30,80,40,.08); border-radius: 10px; color: inherit; }
.preview blockquote { border-left: 3px solid #2ea052; margin: 1em 0; padding: .5em 1em; background: rgba(46,160,82,.04); color: #536751; }
.preview table { border-collapse: collapse; margin: 1em 0; width: 100%; }
.preview th, .preview td { border: 1px solid rgba(30,80,40,.12); padding: 8px 12px; font-size: .85rem; }
.preview th { background: #f0f6ec; font-weight: 600; }
.preview img { max-width: 100%; border-radius: 8px; margin: .5em 0; }
.preview hr { border: none; border-top: 1px solid rgba(30,80,40,.1); margin: 1.5em 0; }
.preview .katex-display { margin: 1em 0; overflow-x: auto; }
/* View modes */
.editor-body.only-editor .preview { display: none; }
.editor-body.only-editor textarea { flex: 1; }
.editor-body.only-preview textarea { display: none; }
.editor-body.only-preview .preview { flex: 1; }

/* ── Empty state ── */
.empty-state { flex: 1; display: flex; align-items: center; justify-content: center; flex-direction: column; color: #8a9b86; }
.empty-state i { font-size: 3rem; margin-bottom: 12px; }
.empty-state p { font-size: .9rem; }

/* ── Modal ── */
.modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,.3); z-index: 200; display: flex; align-items: center; justify-content: center; }
.modal { background: #fff; border-radius: 12px; padding: 24px; width: 400px; max-width: 90vw; }
.modal h3 { font-size: 1rem; margin-bottom: 16px; }
.modal label { display: block; font-size: .82rem; font-weight: 500; margin-bottom: 4px; color: #536751; }
.modal input, .modal select { width: 100%; padding: 8px 12px; border: 1px solid rgba(30,80,40,.15); border-radius: 8px; font-size: .85rem; margin-bottom: 12px; outline: none; }
.modal input:focus, .modal select:focus { border-color: #1f7f3f; }
.modal .modal-actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 8px; }
.modal .modal-actions button { padding: 8px 20px; border-radius: 8px; border: none; cursor: pointer; font-size: .82rem; }
.modal .modal-cancel { background: #eee; color: #536751; }
.modal .modal-ok { background: #1f7f3f; color: #fff; }

/* ── Toast ── */
.toast { position: fixed; bottom: 24px; right: 24px; background: #1f7f3f; color: #fff; padding: 10px 20px; border-radius: 8px; font-size: .85rem; z-index: 300; animation: fadeIn .2s; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
</style>
</head>
<body>

<div class="topbar">
    <div class="logo"><i class="bi bi-flower2"></i> StudyNotes Editor</div>
    <a href="/" class="back"><i class="bi bi-eye"></i> 预览站点</a>
</div>

<div class="wrap">
    <div class="sidebar" id="sidebar"></div>
    <div class="main-panel" id="mainPanel">
        <div class="empty-state">
            <i class="bi bi-pencil-square"></i>
            <p>从左侧选择文章开始编辑</p>
        </div>
    </div>
</div>

<script>
mermaid.initialize({ startOnLoad: false, theme: 'forest' });
const API = '/_admin/api';
let currentPath = null;
let dirty = false;

// ── Load sidebar ──
async function loadTree() {
    const res = await fetch(API + '/content');
    const cats = await res.json();
    const sb = document.getElementById('sidebar');
    let html = '';
    cats.forEach(cat => {
        const artCount = cat.subcategories.reduce((n, s) => n + s.articles.length, 0);
        html += `<div class="cat-group">
            <div class="cat-header" onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display==='none'?'':'none'">
                <span class="dot" style="background:${cat.color}"></span>
                <span>${cat.name}</span>
                <div class="action-row" onclick="event.stopPropagation()">
                    <button class="add-btn" title="新建子分区" onclick="showNewSub('${cat.folder}')"><i class="bi bi-folder-plus"></i></button>
                    <button title="删除分区" onclick="deleteFolder('${cat.folder}','${cat.name}')"><i class="bi bi-trash3"></i></button>
                </div>
                <span class="count">${artCount}</span>
            </div>
            <div class="sub-list">`;
        cat.subcategories.forEach(sub => {
            html += `<div class="sub-group">
                <div class="sub-header">
                    <i class="bi bi-folder2"></i> ${sub.name}
                    <div class="action-row" onclick="event.stopPropagation()">
                        <button class="add-btn" title="新建文章" onclick="showNewArticle('${cat.folder}','${sub.folder}')"><i class="bi bi-file-earmark-plus"></i></button>
                        <button title="删除子分区" onclick="deleteFolder('${cat.folder}/${sub.folder}','${sub.name}')"><i class="bi bi-trash3"></i></button>
                    </div>
                    <span class="count">${sub.articles.length}</span>
                </div>`;
            sub.articles.forEach(art => {
                html += `<a class="art-item" data-path="${art.path}" onclick="openArticle('${art.path}')">
                    <i class="bi bi-file-earmark-text"></i> ${art.title}
                </a>`;
            });
            html += `</div>`;
        });
        html += `</div></div>`;
    });
    html += `<div class="sidebar-bottom"><button onclick="showNewCat()"><i class="bi bi-plus-lg"></i> 新建分区</button></div>`;
    sb.innerHTML = html;
    // Re-highlight current
    if (currentPath) {
        document.querySelectorAll('.art-item').forEach(el => {
            if (el.dataset.path === currentPath) el.classList.add('active');
        });
    }
}

// ── Open article ──
async function openArticle(path) {
    if (dirty && !confirm('有未保存的修改，确定切换？')) return;
    const res = await fetch(API + '/article?path=' + encodeURIComponent(path));
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    currentPath = path;
    dirty = false;
    const title = path.split('/').pop().replace('.md', '');
    document.getElementById('mainPanel').innerHTML = `
        <div class="editor-wrap">
            <div class="editor-toolbar">
                <h2><i class="bi bi-file-earmark-text"></i> ${title}</h2>
                <div style="display:flex;gap:4px;background:rgba(30,80,40,.06);border-radius:8px;padding:2px;">
                    <button class="btn-toggle" id="btnBoth" onclick="setView('both')" title="分栏"><i class="bi bi-layout-split"></i></button>
                    <button class="btn-toggle" id="btnEdit" onclick="setView('edit')" title="纯编辑"><i class="bi bi-code-slash"></i></button>
                    <button class="btn-toggle" id="btnPreview" onclick="setView('preview')" title="纯预览"><i class="bi bi-eye"></i></button>
                </div>
                <button class="btn-upload" onclick="uploadImage()" title="上传图片"><i class="bi bi-image"></i> 图片</button>
                <input type="file" id="imgInput" accept="image/*" style="display:none" onchange="handleImageUpload(this)">
                <button class="btn-delete" onclick="deleteArticle()"><i class="bi bi-trash3"></i> 删除</button>
                <button class="btn-save" id="saveBtn" onclick="saveArticle()"><i class="bi bi-check-lg"></i> 保存</button>
            </div>
            <div class="editor-body" id="editorBody">
                <textarea id="editor" spellcheck="false">${escHtml(data.content)}</textarea>
                <div class="preview markdown-content" id="previewPane"></div>
            </div>
        </div>`;
    const editor = document.getElementById('editor');
    editor.addEventListener('input', () => { dirty = true; renderPreview(); });
    editor.addEventListener('keydown', e => {
        if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); saveArticle(); }
        // Tab inserts spaces
        if (e.key === 'Tab') { e.preventDefault(); const s=editor.selectionStart,en=editor.selectionEnd; editor.value=editor.value.substring(0,s)+'    '+editor.value.substring(en); editor.selectionStart=editor.selectionEnd=s+4; dirty=true; renderPreview(); }
    });
    // Paste image from clipboard
    editor.addEventListener('paste', async (e) => {
        const items = e.clipboardData && e.clipboardData.items;
        if (!items) return;
        for (const item of items) {
            if (item.type.startsWith('image/')) {
                e.preventDefault();
                const file = item.getAsFile();
                const ext = item.type.split('/')[1] || 'png';
                const filename = 'paste_' + Date.now() + '.' + ext;
                try {
                    const resp = await fetch(API + '/upload-image', {
                        method: 'POST',
                        headers: { 'Content-Type': file.type, 'X-Filename': filename },
                        body: file
                    });
                    const data = await resp.json();
                    if (data.ok) {
                        const pos = editor.selectionStart;
                        const tag = '![' + filename + '](' + data.path + ')';
                        editor.value = editor.value.substring(0, pos) + tag + editor.value.substring(editor.selectionEnd);
                        editor.selectionStart = editor.selectionEnd = pos + tag.length;
                        dirty = true;
                        renderPreview();
                        toast('图片已上传');
                    }
                } catch(err) { toast('图片上传失败: ' + err.message); }
                break;
            }
        }
    });
    // Sync scroll
    editor.addEventListener('scroll', () => {
        const pane = document.getElementById('previewPane');
        if (!pane) return;
        const pct = editor.scrollTop / (editor.scrollHeight - editor.clientHeight || 1);
        pane.scrollTop = pct * (pane.scrollHeight - pane.clientHeight);
    });
    setView('both');
    renderPreview();
    // Highlight sidebar
    document.querySelectorAll('.art-item').forEach(el => el.classList.remove('active'));
    const active = document.querySelector(`.art-item[data-path="${path}"]`);
    if (active) active.classList.add('active');
}

function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

function uploadImage() { document.getElementById('imgInput').click(); }
async function handleImageUpload(input) {
    const file = input.files[0];
    if (!file) return;
    try {
        const resp = await fetch(API + '/upload-image', {
            method: 'POST',
            headers: { 'Content-Type': file.type, 'X-Filename': file.name },
            body: file
        });
        const data = await resp.json();
        if (data.ok) {
            const editor = document.getElementById('editor');
            const pos = editor.selectionStart;
            const tag = `![${file.name}](${data.path})`;
            editor.value = editor.value.substring(0, pos) + tag + editor.value.substring(editor.selectionEnd);
            editor.selectionStart = editor.selectionEnd = pos + tag.length;
            dirty = true;
            renderPreview();
        } else {
            alert('上传失败: ' + (data.error || '未知错误'));
        }
    } catch(e) { alert('上传出错: ' + e.message); }
    input.value = '';
}
async function saveArticle() {
    if (!currentPath) return;
    const content = document.getElementById('editor').value;
    const btn = document.getElementById('saveBtn');
    btn.disabled = true; btn.textContent = '保存中...';
    await fetch(API + '/save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ path: currentPath, content })
    });
    dirty = false;
    btn.disabled = false; btn.innerHTML = '<i class="bi bi-check-lg"></i> 保存';
    toast('已保存并重新构建');
    loadTree();
}

async function deleteArticle() {
    if (!currentPath || !confirm('确定删除这篇文章？')) return;
    await fetch(API + '/delete', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ path: currentPath, type: 'article' })
    });
    currentPath = null; dirty = false;
    document.getElementById('mainPanel').innerHTML = `<div class="empty-state"><i class="bi bi-pencil-square"></i><p>从左侧选择文章开始编辑</p></div>`;
    toast('已删除');
    loadTree();
}

async function deleteFolder(path, name) {
    if (!confirm(`确定删除「${name}」及其所有内容？`)) return;
    await fetch(API + '/delete', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ path, type: 'folder' })
    });
    if (currentPath && currentPath.startsWith(path + '/')) {
        currentPath = null; dirty = false;
        document.getElementById('mainPanel').innerHTML = `<div class="empty-state"><i class="bi bi-pencil-square"></i><p>从左侧选择文章开始编辑</p></div>`;
    }
    toast('已删除');
    loadTree();
}

let _renderTimer = null;
function renderPreview() {
    clearTimeout(_renderTimer);
    _renderTimer = setTimeout(_doRender, 150);
}
function _doRender() {
    const pane = document.getElementById('previewPane');
    const editor = document.getElementById('editor');
    if (!pane || !editor) return;
    let md = editor.value;
    // Strip YAML front matter for preview
    if (md.startsWith('---')) {
        const end = md.indexOf('---', 3);
        if (end > 0) md = md.substring(end + 3).trim();
    }
    // Rewrite image paths for local preview
    md = md.replace(/!\[\[([^\]]+\.(?:png|jpg|jpeg|gif|svg|webp))\]\]/gi, (m, p) => {
        const alt = p.split('/').pop();
        return `![${alt}](/static/images/${p})`;
    });
    md = md.replace(/\/media\/images\//g, '/static/images/');
    // Protect math from marked.js
    const mathBlocks = [], mathInlines = [];
    md = md.replace(/\$\$([\s\S]+?)\$\$/g, (m) => { mathBlocks.push(m); return '\x00MBLOCK_'+(mathBlocks.length-1)+'\x00'; });
    md = md.replace(/\\\[([\s\S]+?)\\\]/g, (m) => { mathBlocks.push(m); return '\x00MBLOCK_'+(mathBlocks.length-1)+'\x00'; });
    md = md.replace(/(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)/g, (m) => { mathInlines.push(m); return '\x00MINLINE_'+(mathInlines.length-1)+'\x00'; });
    md = md.replace(/\\\((.+?)\\\)/g, (m) => { mathInlines.push(m); return '\x00MINLINE_'+(mathInlines.length-1)+'\x00'; });
    // Render markdown
    let html = marked.parse(md, { breaks: false, gfm: true, highlight: (code, lang) => {
        if (lang && hljs.getLanguage(lang)) return hljs.highlight(code, {language: lang}).value;
        return hljs.highlightAuto(code).value;
    }});
    // Restore math
    mathBlocks.forEach((m, i) => { html = html.replace('\x00MBLOCK_'+i+'\x00', m); });
    mathInlines.forEach((m, i) => { html = html.replace('\x00MINLINE_'+i+'\x00', m); });
    pane.innerHTML = html;
    // Render KaTeX
    if (typeof renderMathInElement !== 'undefined') {
        renderMathInElement(pane, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false},
                {left: '\\[', right: '\\]', display: true},
                {left: '\\(', right: '\\)', display: false}
            ],
            throwOnError: false
        });
    }
    // Render Mermaid diagrams
    if (typeof mermaid !== 'undefined') {
        pane.querySelectorAll('pre code.language-mermaid').forEach(function(code) {
            var pre = code.parentElement;
            var div = document.createElement('div');
            div.className = 'mermaid';
            div.textContent = code.textContent;
            pre.parentNode.replaceChild(div, pre);
        });
        if (pane.querySelector('.mermaid')) {
            mermaid.run({ nodes: pane.querySelectorAll('.mermaid') });
        }
    }
}

let currentView = 'both';
function setView(mode) {
    currentView = mode;
    const body = document.getElementById('editorBody');
    body.classList.remove('only-editor', 'only-preview');
    if (mode === 'edit') body.classList.add('only-editor');
    if (mode === 'preview') body.classList.add('only-preview');
    document.querySelectorAll('.btn-toggle').forEach(b => b.classList.remove('active'));
    const btn = {both:'btnBoth', edit:'btnEdit', preview:'btnPreview'}[mode];
    document.getElementById(btn)?.classList.add('active');
    if (mode !== 'edit') renderPreview();
}

// ── Modals ──
function showModal(title, fields, onOk) {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    let fieldsHtml = fields.map(f => {
        if (f.type === 'select') {
            const opts = f.options.map(o => `<option value="${o.value}">${o.label}</option>`).join('');
            return `<label>${f.label}</label><select id="modal_${f.key}">${opts}</select>`;
        }
        return `<label>${f.label}</label><input id="modal_${f.key}" type="text" value="${f.default || ''}" placeholder="${f.placeholder || ''}">`;
    }).join('');
    overlay.innerHTML = `<div class="modal"><h3>${title}</h3>${fieldsHtml}<div class="modal-actions"><button class="modal-cancel" onclick="this.closest('.modal-overlay').remove()">取消</button><button class="modal-ok" id="modalOk">确定</button></div></div>`;
    document.body.appendChild(overlay);
    overlay.querySelector('.modal-cancel').onclick = () => overlay.remove();
    overlay.querySelector('#modalOk').onclick = () => {
        const vals = {};
        fields.forEach(f => { vals[f.key] = document.getElementById('modal_' + f.key).value; });
        overlay.remove();
        onOk(vals);
    };
    overlay.querySelector('input,select')?.focus();
}

function showNewCat() {
    showModal('新建分区', [
        { key: 'name', label: '分区名称', placeholder: '如：机器学习' },
        { key: 'icon', label: '图标（Bootstrap Icons）', default: 'bi-folder' },
        { key: 'color', label: '主题色', default: '#2ea052' },
        { key: 'desc', label: '描述', placeholder: '可选' },
    ], async vals => {
        await fetch(API + '/new-folder', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ path: vals.name, meta: { name: vals.name, icon: vals.icon, color: vals.color, description: vals.desc, order: 0 } })
        });
        toast('分区已创建');
        loadTree();
    });
}

function showNewSub(catFolder) {
    showModal('新建子分区', [
        { key: 'name', label: '子分区名称', placeholder: '如：基础概念' },
    ], async vals => {
        await fetch(API + '/new-folder', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ path: catFolder + '/' + vals.name, meta: { name: vals.name, order: 0 } })
        });
        toast('子分区已创建');
        loadTree();
    });
}

function showNewArticle(catFolder, subFolder) {
    showModal('新建文章', [
        { key: 'title', label: '文章标题', placeholder: '如：线性回归' },
    ], async vals => {
        const res = await fetch(API + '/new-article', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ category: catFolder, subcategory: subFolder, title: vals.title })
        });
        const data = await res.json();
        toast('文章已创建');
        await loadTree();
        if (data.path) openArticle(data.path);
    });
}

function toast(msg) {
    const t = document.createElement('div');
    t.className = 'toast';
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 2000);
}

// ── Init ──
window.addEventListener('beforeunload', e => { if (dirty) { e.preventDefault(); e.returnValue = ''; } });
loadTree();
</script>
</body>
</html>
'''


# ── Watcher ────────────────────────────────────────────────
def get_mtime_map():
    mtimes = {}
    for d in WATCH_DIRS:
        if not d.exists():
            continue
        for f in d.rglob('*'):
            if f.is_file():
                mtimes[f] = f.stat().st_mtime
    return mtimes


def watcher():
    last = get_mtime_map()
    while True:
        time.sleep(1.5)
        current = get_mtime_map()
        if current != last:
            last = current


# ── Main ───────────────────────────────────────────────────
def main():
    print('Building site...')
    rebuild()
    print(f'\n  Site:   http://127.0.0.1:{PORT}/')
    print(f'  Editor: http://127.0.0.1:{PORT}/_admin/')
    print('  Ctrl+C to stop\n')

    t = threading.Thread(target=watcher, daemon=True)
    t.start()

    server = HTTPServer(('127.0.0.1', PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
