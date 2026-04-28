#!/usr/bin/env python3
"""
StudyNotes — static site generator.

Reads markdown files from content/, renders them with Jinja2 templates,
and writes the resulting HTML to docs/ for GitHub Pages.

Directory layout:
    content/
      分类名/
        _meta.yml          (name, description, icon, color, order)
        子分类名/
          _meta.yml        (name, order)
          文章标题.md       (YAML front-matter + markdown body)
"""

import os
import re
import shutil
import math
import yaml
import markdown as md_lib
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

# ── Paths ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
CONTENT_DIR = ROOT / 'content'
TEMPLATE_DIR = ROOT / 'templates'
STATIC_DIR = ROOT / 'static'
OUTPUT_DIR = ROOT / 'docs'

# ── Markdown helpers (ported from Django templatetags) ─────
_MATH_BLOCK_PH = '\x00MATHBLOCK_%d\x00'
_MATH_INLINE_PH = '\x00MATHINLINE_%d\x00'


def _protect_math(text):
    blocks, inlines = [], []

    def save_block(m):
        blocks.append(m.group(0))
        return _MATH_BLOCK_PH % (len(blocks) - 1)

    def save_inline(m):
        inlines.append(m.group(0))
        return _MATH_INLINE_PH % (len(inlines) - 1)

    text = re.sub(r'\$\$(.+?)\$\$', save_block, text, flags=re.DOTALL)
    text = re.sub(r'\\\[(.+?)\\\]', save_block, text, flags=re.DOTALL)
    text = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', save_inline, text)
    text = re.sub(r'\\\((.+?)\\\)', save_inline, text)
    return text, blocks, inlines


def _restore_math(html, blocks, inlines):
    for i, b in enumerate(blocks):
        html = html.replace(_MATH_BLOCK_PH % i, b)
    for i, b in enumerate(inlines):
        html = html.replace(_MATH_INLINE_PH % i, b)
    return html


def convert_obsidian_images(text):
    def replace_image(m):
        path = m.group(1).strip()
        alt = path.split('/')[-1]
        return f'![{alt}](static/images/{path})'
    return re.sub(r'!\[\[([^\]]+\.(?:png|jpg|jpeg|gif|svg|webp))\]\]', replace_image, text, flags=re.IGNORECASE)


def convert_obsidian_links(text):
    return re.sub(r'\[\[([^\]|]+?)(?:\|([^\]]+?))?\]\]', lambda m: m.group(2) or m.group(1), text)


def _ensure_blank_before_blocks(text):
    text = re.sub(r'(\S[^\n]*)\n([ \t]*[-*+] )', r'\1\n\n\2', text)
    text = re.sub(r'(\S[^\n]*)\n([ \t]*\d+\. )', r'\1\n\n\2', text)
    text = re.sub(r'(\S[^\n]*)\n(>[ \t])', r'\1\n\n\2', text)
    return text


def add_heading_ids(html):
    counter = {}
    def replace_heading(m):
        level, content = m.group(1), m.group(2)
        slug = re.sub(r'[^\w\u4e00-\u9fff-]', '-', content.strip()).strip('-').lower()
        if not slug:
            slug = 'heading'
        counter[slug] = counter.get(slug, 0) + 1
        if counter[slug] > 1:
            slug = f'{slug}-{counter[slug]}'
        return f'<h{level} id="{slug}">{content}</h{level}>'
    return re.sub(r'<h([1-6])>(.*?)</h\1>', replace_heading, html)


def render_markdown(text, root=''):
    if not text:
        return ''
    # Rewrite legacy Django media paths to static
    text = text.replace('/media/images/', f'{root}static/images/')
    text = convert_obsidian_images(text)
    text = convert_obsidian_links(text)
    text = _ensure_blank_before_blocks(text)
    text, math_blocks, math_inlines = _protect_math(text)

    converter = md_lib.Markdown(extensions=[
        'markdown.extensions.fenced_code',
        'markdown.extensions.codehilite',
        'markdown.extensions.tables',
        'markdown.extensions.toc',
        'markdown.extensions.sane_lists',
        'markdown.extensions.attr_list',
    ], extension_configs={
        'markdown.extensions.codehilite': {
            'css_class': 'highlight',
            'linenums': False,
            'guess_lang': True,
        },
    })
    html = converter.convert(text)
    html = add_heading_ids(html)
    html = _restore_math(html, math_blocks, math_inlines)
    return html


def generate_toc(html_content):
    headings = re.findall(r'<h([1-4])\s*id="([^"]*)"[^>]*>(.*?)</h\1>', html_content, re.DOTALL)
    if not headings:
        return ''
    toc = '<nav class="toc-nav"><ul class="toc-list">'
    for level, anchor_id, title in headings:
        clean_title = re.sub(r'<[^>]+>', '', title)
        toc += f'<li class="toc-item toc-level-{level}"><a href="#{anchor_id}">{clean_title}</a></li>'
    toc += '</ul></nav>'
    return toc


# ── Content loader ─────────────────────────────────────────
def load_meta(path):
    meta_file = path / '_meta.yml'
    if meta_file.exists():
        with open(meta_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_article(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = f.read()
    # Parse YAML front matter
    if raw.startswith('---'):
        parts = raw.split('---', 2)
        if len(parts) >= 3:
            front = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()
        else:
            front, body = {}, raw
    else:
        front, body = {}, raw

    title = front.get('title', filepath.stem)
    return {
        'title': title,
        'summary': front.get('summary', ''),
        'order': front.get('order', 0),
        'pinned': front.get('pinned', False),
        'created': front.get('created', ''),
        'updated': front.get('updated', ''),
        'content_raw': body,
        'slug': re.sub(r'[^\w\u4e00-\u9fff-]', '-', title).strip('-'),
        'filename': filepath.stem,
    }


def load_content():
    categories = []
    for cat_path in sorted(CONTENT_DIR.iterdir()):
        if not cat_path.is_dir():
            continue
        cat_meta = load_meta(cat_path)
        cat = {
            'name': cat_meta.get('name', cat_path.name),
            'description': cat_meta.get('description', ''),
            'icon': cat_meta.get('icon', 'bi-folder'),
            'color': cat_meta.get('color', '#2ea052'),
            'order': cat_meta.get('order', 0),
            'slug': re.sub(r'[^\w\u4e00-\u9fff-]', '-', cat_meta.get('name', cat_path.name)).strip('-'),
            'subcategories': [],
            'article_count': 0,
        }

        for sub_path in sorted(cat_path.iterdir()):
            if not sub_path.is_dir():
                continue
            sub_meta = load_meta(sub_path)
            sub = {
                'name': sub_meta.get('name', sub_path.name),
                'order': sub_meta.get('order', 0),
                'slug': re.sub(r'[^\w\u4e00-\u9fff-]', '-', sub_meta.get('name', sub_path.name)).strip('-'),
                'articles': [],
            }

            for md_file in sorted(sub_path.glob('*.md')):
                article = parse_article(md_file)
                sub['articles'].append(article)

            # Sort: pinned first, then by order, then by title
            sub['articles'].sort(key=lambda a: (not a['pinned'], a['order'], a['title']))
            cat['subcategories'].append(sub)
            cat['article_count'] += len(sub['articles'])

        cat['subcategories'].sort(key=lambda s: (s['order'], s['name']))
        categories.append(cat)

    categories.sort(key=lambda c: (c['order'], c['name']))

    total_articles = sum(c['article_count'] for c in categories)
    return categories, total_articles


# ── Build ──────────────────────────────────────────────────
def build():
    categories, total_articles = load_content()

    # Clean output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # Copy static assets (includes images)
    shutil.copytree(STATIC_DIR, OUTPUT_DIR / 'static')

    # Set up Jinja2
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=False)
    env.globals['total_categories'] = len(categories)
    env.globals['total_articles'] = total_articles
    env.globals['categories'] = categories

    # Helper: compute width ratio for overview bars
    def widthratio(value, max_value):
        if not max_value:
            return 0
        return round(value / max_value * 100)
    env.globals['widthratio'] = widthratio

    # ── Home page ──
    tpl = env.get_template('home.html')
    # Recent articles: collect all, sort by updated, take 8
    all_articles = []
    for cat in categories:
        for sub in cat['subcategories']:
            for art in sub['articles']:
                art['_cat'] = cat
                art['_sub'] = sub
                all_articles.append(art)

    recent = sorted(all_articles, key=lambda a: a.get('updated', ''), reverse=True)[:8]

    # root = '' for home (top-level)
    html = tpl.render(categories=categories, recent_articles=recent, root='')
    (OUTPUT_DIR / 'index.html').write_text(html, encoding='utf-8')

    # ── Category pages ──  (depth 1 → root = '../')
    cat_tpl = env.get_template('category_detail.html')
    for cat in categories:
        cat_dir = OUTPUT_DIR / cat['slug']
        cat_dir.mkdir(parents=True, exist_ok=True)
        html = cat_tpl.render(category=cat, root='../')
        (cat_dir / 'index.html').write_text(html, encoding='utf-8')

        # ── Article pages ──  (depth 3 → root = '../../../')
        art_tpl = env.get_template('article_detail.html')
        for sub in cat['subcategories']:
            for art in sub['articles']:
                rendered_content = render_markdown(art['content_raw'], root='../../../')
                toc_html = generate_toc(rendered_content)

                # Siblings in same subcategory
                siblings = [a for a in sub['articles'] if a['title'] != art['title']]

                art_dir = cat_dir / sub['slug'] / art['slug']
                art_dir.mkdir(parents=True, exist_ok=True)

                html = art_tpl.render(
                    article=art,
                    article_content=rendered_content,
                    article_toc=toc_html,
                    category=cat,
                    subcategory=sub,
                    siblings=siblings,
                    root='../../../',
                )
                (art_dir / 'index.html').write_text(html, encoding='utf-8')

    # ── .nojekyll (tell GitHub Pages not to process with Jekyll) ──
    (OUTPUT_DIR / '.nojekyll').touch()

    print(f'Built {len(categories)} categories, {total_articles} articles → docs/')


if __name__ == '__main__':
    build()
