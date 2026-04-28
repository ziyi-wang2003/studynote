import re
import markdown
from django import template
from django.utils.safestring import mark_safe

register = template.Library()

# Placeholder for math blocks during markdown processing
_MATH_BLOCK_PH = '\x00MATHBLOCK_%d\x00'
_MATH_INLINE_PH = '\x00MATHINLINE_%d\x00'


def _protect_math(text):
    """Extract math expressions before markdown processing to prevent mangling."""
    blocks = []
    inlines = []

    # Protect $$ ... $$ block math (including multiline)
    def save_block(m):
        blocks.append(m.group(0))
        return _MATH_BLOCK_PH % (len(blocks) - 1)
    text = re.sub(r'\$\$(.+?)\$\$', save_block, text, flags=re.DOTALL)

    # Protect \[ ... \] block math (including multiline)
    text = re.sub(r'\\\[(.+?)\\\]', save_block, text, flags=re.DOTALL)

    # Protect $ ... $ inline math (single line only, non-greedy)
    def save_inline(m):
        inlines.append(m.group(0))
        return _MATH_INLINE_PH % (len(inlines) - 1)
    text = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', save_inline, text)

    # Protect \( ... \) inline math
    text = re.sub(r'\\\((.+?)\\\)', save_inline, text)

    return text, blocks, inlines


def _restore_math(html, blocks, inlines):
    """Put math expressions back after markdown processing."""
    for i, b in enumerate(blocks):
        html = html.replace(_MATH_BLOCK_PH % i, b)
    for i, b in enumerate(inlines):
        html = html.replace(_MATH_INLINE_PH % i, b)
    return html


def convert_obsidian_images(text):
    """Convert Obsidian ![[image]] syntax to standard markdown images.

    Supports patterns like:
        ![[大模型知识点/后训练算法/others/Pasted image 20260406175237.png]]
    Converts to:
        ![Pasted image 20260406175237.png](/media/images/大模型知识点/后训练算法/others/Pasted image 20260406175237.png)
    """
    def replace_image(match):
        path = match.group(1).strip()
        alt = path.split('/')[-1]
        return f'![{alt}](/media/images/{path})'
    return re.sub(r'!\[\[([^\]]+\.(?:png|jpg|jpeg|gif|svg|webp))\]\]', replace_image, text, flags=re.IGNORECASE)


def convert_obsidian_links(text):
    """Convert Obsidian [[link]] syntax to plain text (non-image)."""
    return re.sub(r'\[\[([^\]|]+?)(?:\|([^\]]+?))?\]\]', lambda m: m.group(2) or m.group(1), text)


def _ensure_blank_before_blocks(text):
    """Ensure a blank line before list items and other block elements.

    Python-Markdown requires a blank line before lists, otherwise they
    are treated as continuation of the preceding paragraph.
    """
    # Insert blank line before unordered list items (- or * or +) when preceded by non-blank, non-list line
    text = re.sub(r'(\S[^\n]*)\n([ \t]*[-*+] )', r'\1\n\n\2', text)
    # Insert blank line before ordered list items (1. 2. etc.)
    text = re.sub(r'(\S[^\n]*)\n([ \t]*\d+\. )', r'\1\n\n\2', text)
    # Insert blank line before blockquotes
    text = re.sub(r'(\S[^\n]*)\n(>[ \t])', r'\1\n\n\2', text)
    return text


def generate_toc(html_content):
    """Extract headings from HTML and generate a table of contents."""
    headings = re.findall(r'<h([1-4])\s*id="([^"]*)"[^>]*>(.*?)</h\1>', html_content, re.DOTALL)
    if not headings:
        return ''

    toc = '<nav class="toc-nav"><ul class="toc-list">'
    for level, anchor_id, title in headings:
        clean_title = re.sub(r'<[^>]+>', '', title)
        toc += f'<li class="toc-item toc-level-{level}"><a href="#{anchor_id}">{clean_title}</a></li>'
    toc += '</ul></nav>'
    return toc


def add_heading_ids(text):
    """Add id attributes to markdown headings for anchor links."""
    counter = {}

    def replace_heading(match):
        level = match.group(1)
        content = match.group(2)
        slug = re.sub(r'[^\w\u4e00-\u9fff-]', '-', content.strip()).strip('-').lower()
        if not slug:
            slug = 'heading'
        counter[slug] = counter.get(slug, 0) + 1
        if counter[slug] > 1:
            slug = f'{slug}-{counter[slug]}'
        return f'<h{level} id="{slug}">{content}</h{level}>'

    return re.sub(r'<h([1-6])>(.*?)</h\1>', replace_heading, text)


@register.filter(name='markdown')
def markdown_filter(text):
    """Render markdown text to HTML with extensions."""
    if not text:
        return ''

    text = convert_obsidian_images(text)
    text = convert_obsidian_links(text)
    text = _ensure_blank_before_blocks(text)

    # Protect math from markdown mangling
    text, math_blocks, math_inlines = _protect_math(text)

    md = markdown.Markdown(extensions=[
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

    html = md.convert(text)
    html = add_heading_ids(html)
    html = _restore_math(html, math_blocks, math_inlines)
    return mark_safe(html)


@register.filter(name='markdown_toc')
def markdown_toc_filter(text):
    """Generate TOC from markdown text."""
    if not text:
        return ''
    text = convert_obsidian_images(text)
    text = convert_obsidian_links(text)
    text, math_blocks, math_inlines = _protect_math(text)

    md = markdown.Markdown(extensions=[
        'markdown.extensions.fenced_code',
        'markdown.extensions.tables',
        'markdown.extensions.toc',
    ])
    html = md.convert(text)
    html = add_heading_ids(html)
    html = _restore_math(html, math_blocks, math_inlines)
    return mark_safe(generate_toc(html))
