"""Markdown → email-safe HTML (inline CSS, renders in Gmail/iCloud Mail).

Lifted from projects/world-understanding/scripts/md_to_html.py — the proven
converter the daily-report task uses — so framework emails (pipeline
completion notices) share one implementation instead of sending raw markdown.
Handles headers, tables, ordered/unordered lists, bold/italic/strike, code
spans, links, and horizontal rules.
"""
import re

def inline_md(text):
    """Convert inline markdown: **bold**, *italic*, `code`, links."""
    text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\w)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    text = re.sub(r'`(.+?)`', r'<code style="background:#f0f0f0;padding:1px 4px;border-radius:3px;">\1</code>', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    return text

def md_to_html(md_text):
    """Convert markdown to HTML email."""
    lines = md_text.split('\n')
    html_parts = []
    in_table = False
    in_ul = False
    in_ol = False
    table_rows = []

    def close_list():
        nonlocal in_ul, in_ol
        if in_ul:
            html_parts.append('</ul>')
            in_ul = False
        if in_ol:
            html_parts.append('</ol>')
            in_ol = False

    def close_table():
        nonlocal in_table, table_rows
        if in_table and table_rows:
            html_parts.append('<table>')
            first_data = True
            for row in table_rows:
                cells = [c.strip() for c in row.split('|')[1:-1]]
                if all(c.startswith('---') for c in cells):
                    continue
                tag = 'th' if first_data else 'td'
                first_data = False
                row_html = '<tr>'
                for c in cells:
                    row_html += '<' + tag + '>' + inline_md(c) + '</' + tag + '>'
                row_html += '</tr>'
                html_parts.append(row_html)
            html_parts.append('</table>')
        in_table = False
        table_rows = []

    for line in lines:
        stripped = line.strip()

        # Headers
        if stripped.startswith('### '):
            close_list()
            close_table()
            html_parts.append('<h3>' + inline_md(stripped[4:]) + '</h3>')
            continue
        if stripped.startswith('## '):
            close_list()
            close_table()
            html_parts.append('<h2>' + inline_md(stripped[3:]) + '</h2>')
            continue
        if stripped.startswith('# '):
            close_list()
            close_table()
            html_parts.append('<h1>' + inline_md(stripped[2:]) + '</h1>')
            continue

        # Horizontal rule
        if stripped.startswith('---'):
            close_list()
            close_table()
            html_parts.append('<hr>')
            continue

        # Tables
        if '|' in stripped and stripped.endswith('|'):
            close_list()
            if not in_table:
                in_table = True
                table_rows = []
            table_rows.append(stripped)
            continue

        close_table()

        # Unordered list
        if stripped.startswith('- ') or stripped.startswith('* '):
            if not in_ul:
                in_ul = True
                html_parts.append('<ul>')
            html_parts.append('<li>' + inline_md(stripped[2:]) + '</li>')
            continue

        # Ordered list
        m = re.match(r'^\d+\.\s(.*)', stripped)
        if m:
            close_list()
            if not in_ol:
                in_ol = True
                html_parts.append('<ol>')
            html_parts.append('<li>' + inline_md(m.group(1)) + '</li>')
            continue

        # Empty line
        if stripped == '':
            close_list()
            continue

        # Paragraph
        close_list()
        html_parts.append('<p>' + inline_md(stripped) + '</p>')

    close_list()
    close_table()

    body = '\n'.join(html_parts)
    css = (
        'body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; '
        'line-height: 1.6; color: #333; max-width: 720px; margin: 0 auto; padding: 24px; } '
        'h1 { font-size: 24px; color: #111; border-bottom: 2px solid #eee; padding-bottom: 8px; } '
        'h2 { font-size: 20px; color: #222; margin-top: 24px; } '
        'h3 { font-size: 17px; color: #333; } '
        'table { border-collapse: collapse; width: 100%; margin: 12px 0; } '
        'th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; } '
        'th { background: #f5f5f5; font-weight: 600; } '
        'ul, ol { margin: 8px 0; padding-left: 24px; } '
        'li { margin: 4px 0; } '
        'code { background: #f0f0f0; padding: 1px 5px; border-radius: 3px; font-size: 14px; } '
        'hr { border: none; border-top: 1px solid #ddd; margin: 16px 0; } '
        'a { color: #1a73e8; text-decoration: none; }'
    )
    return '<!DOCTYPE html>\n<html><head><meta charset="utf-8"><style>' + css + '</style></head><body>' + body + '</body></html>'
