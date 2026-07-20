"""Minimal flowable-DSL -> PDF renderer built on reportlab.

Content lives in `doc_content.py` as a flat list of (kind, payload) tuples.
This module only knows how to turn those into a paginated, bookmarked PDF
with a table of contents.
"""
import os
import re
import glob

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageBreak, PageTemplate,
                                Paragraph, Preformatted, Spacer, Table, TableStyle,
                                KeepTogether, CondPageBreak)
from reportlab.platypus.tableofcontents import TableOfContents


# ----------------------------------------------------------------- fonts
_FDIR = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')

def register_fonts():
    reg = [
        ('DejaVu',        'DejaVuSans.ttf'),
        ('DejaVu-Bold',   'DejaVuSans-Bold.ttf'),
        ('DejaVu-Italic', 'DejaVuSans-Oblique.ttf'),
        ('DejaVu-BoldItalic', 'DejaVuSans-BoldOblique.ttf'),
        ('Mono',          'DejaVuSansMono.ttf'),
        ('Mono-Bold',     'DejaVuSansMono-Bold.ttf'),
        ('Mono-Italic',   'DejaVuSansMono-Oblique.ttf'),
    ]
    for name, fn in reg:
        pdfmetrics.registerFont(TTFont(name, os.path.join(_FDIR, fn)))
    pdfmetrics.registerFontFamily(
        'DejaVu', normal='DejaVu', bold='DejaVu-Bold',
        italic='DejaVu-Italic', boldItalic='DejaVu-BoldItalic')
    pdfmetrics.registerFontFamily(
        'Mono', normal='Mono', bold='Mono-Bold', italic='Mono-Italic')


# ----------------------------------------------------------------- colours
INK      = colors.HexColor('#1a1a1a')
MUTED    = colors.HexColor('#5a5a5a')
ACCENT   = colors.HexColor('#0b4f6c')
ACCENT2  = colors.HexColor('#7a3b2e')
CODE_BG  = colors.HexColor('#f4f4f2')
CODE_BR  = colors.HexColor('#d8d8d4')
NOTE_BG  = colors.HexColor('#fdf6e3')
NOTE_BR  = colors.HexColor('#e0d5b0')
WARN_BG  = colors.HexColor('#fdeeec')
WARN_BR  = colors.HexColor('#e8c4bd')
MATH_BG  = colors.HexColor('#f7f9fb')

# Lines per un-splittable code sub-row. Must be small enough that one row
# always fits a page (~62 mono lines at 10.4pt leading in a 260mm frame).
CODE_ROWS = 20


# ----------------------------------------------------------------- inline markup
def _esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def md(s):
    """Escape XML, then apply `code`, **bold**, *italic*.

    Code spans are lifted out before the emphasis passes run: a literal `*`
    inside a code span (e.g. `*.npz`) would otherwise be eaten by the italic
    regex and emit crossed tags like <font><i></font></i>.
    """
    s = _esc(s)

    spans = []

    def _stash(m):
        spans.append(m.group(1))
        return f'\x00{len(spans) - 1}\x00'

    s = re.sub(r'`([^`]+)`', _stash, s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'(?<![\w*])\*([^*\n]+)\*(?![\w*])', r'<i>\1</i>', s)

    def _restore(m):
        body = spans[int(m.group(1))]
        return f'<font face="Mono" size="8.6" color="#7a3b2e">{body}</font>'

    return re.sub(r'\x00(\d+)\x00', _restore, s)


def plain(s):
    """Strip the inline markup. For bookmarks, the TOC, and running headers."""
    return s.replace('`', '').replace('**', '').replace('*', '')


# ----------------------------------------------------------------- styles
def build_styles():
    ss = getSampleStyleSheet()

    body = ParagraphStyle(
        'Body', parent=ss['BodyText'], fontName='DejaVu', fontSize=9.4,
        leading=14.2, alignment=TA_JUSTIFY, textColor=INK,
        spaceBefore=0, spaceAfter=7)

    h1 = ParagraphStyle(
        'H1', fontName='DejaVu-Bold', fontSize=19, leading=23, textColor=ACCENT,
        spaceBefore=0, spaceAfter=12)
    h2 = ParagraphStyle(
        'H2', fontName='DejaVu-Bold', fontSize=13.5, leading=17, textColor=ACCENT,
        spaceBefore=15, spaceAfter=7)
    h3 = ParagraphStyle(
        'H3', fontName='DejaVu-Bold', fontSize=10.8, leading=14, textColor=ACCENT2,
        spaceBefore=11, spaceAfter=5)

    # Border/background come from the wrapping Table in build_flowables().
    code = ParagraphStyle(
        'Code', fontName='Mono', fontSize=7.9, leading=10.4, textColor=INK,
        spaceBefore=0, spaceAfter=0)

    math = ParagraphStyle(
        'Math', fontName='Mono', fontSize=8.8, leading=13.6, textColor=INK,
        backColor=MATH_BG, alignment=TA_CENTER, borderPadding=(7, 5, 7, 5),
        spaceBefore=5, spaceAfter=10)

    bullet = ParagraphStyle(
        'Bullet', parent=body, leftIndent=13, bulletIndent=3,
        spaceAfter=3.5, alignment=TA_JUSTIFY)

    note = ParagraphStyle(
        'Note', parent=body, backColor=NOTE_BG, borderColor=NOTE_BR,
        borderWidth=0.6, borderPadding=(7, 7, 7, 7), spaceBefore=5,
        spaceAfter=10, alignment=TA_JUSTIFY)
    warn = ParagraphStyle(
        'Warn', parent=note, backColor=WARN_BG, borderColor=WARN_BR)

    cap = ParagraphStyle(
        'Caption', parent=body, fontSize=8.2, leading=11, textColor=MUTED,
        alignment=TA_CENTER, spaceBefore=-4, spaceAfter=10)

    cell = ParagraphStyle('Cell', fontName='DejaVu', fontSize=8.0, leading=10.8,
                          textColor=INK)
    cellh = ParagraphStyle('CellH', parent=cell, fontName='DejaVu-Bold',
                           textColor=colors.white)
    cellm = ParagraphStyle('CellM', parent=cell, fontName='Mono', fontSize=7.5,
                           leading=10.2)

    toc1 = ParagraphStyle('TOC1', fontName='DejaVu-Bold', fontSize=10, leading=17,
                          textColor=ACCENT)
    toc2 = ParagraphStyle('TOC2', fontName='DejaVu', fontSize=9, leading=14,
                          leftIndent=14, textColor=INK)
    toc3 = ParagraphStyle('TOC3', fontName='DejaVu', fontSize=8.3, leading=12,
                          leftIndent=30, textColor=MUTED)

    # 20pt keeps the 30-char module name on one line in a 168 mm frame.
    title = ParagraphStyle('Title', fontName='DejaVu-Bold', fontSize=20,
                           leading=26, textColor=ACCENT, alignment=TA_CENTER,
                           spaceAfter=10)
    subtitle = ParagraphStyle('Sub', fontName='DejaVu', fontSize=12,
                              leading=17, textColor=MUTED, alignment=TA_CENTER,
                              spaceAfter=6)

    return dict(body=body, h1=h1, h2=h2, h3=h3, code=code, math=math,
                bullet=bullet, note=note, warn=warn, cap=cap, cell=cell,
                cellh=cellh, cellm=cellm, toc=[toc1, toc2, toc3],
                title=title, subtitle=subtitle)


# ----------------------------------------------------------------- doc template
class Doc(BaseDocTemplate):
    def __init__(self, path, **kw):
        super().__init__(path, pagesize=A4,
                         leftMargin=20 * mm, rightMargin=18 * mm,
                         topMargin=18 * mm, bottomMargin=17 * mm, **kw)
        frame = Frame(self.leftMargin, self.bottomMargin,
                      self.width, self.height, id='body',
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        # onPageEnd, not onPage: `_h1` is updated by afterFlowable as headings
        # are laid out, so at page *start* it still names the previous section.
        self.addPageTemplates([
            PageTemplate(id='plain', frames=[frame], onPage=self._decorate_plain),
            PageTemplate(id='body', frames=[frame], onPageEnd=self._decorate),
        ])
        self._h1 = ''
        self._seq = 0

    def beforeDocument(self):
        # multiBuild runs this story several times; the bookmark keys must be
        # identical on every pass or the TOC never compares equal and the
        # build loops until `Index entries not resolved`.
        self._seq = 0
        self._h1 = ''

    def _decorate_plain(self, canv, doc):
        pass

    def _decorate(self, canv, doc):
        canv.saveState()
        canv.setFont('DejaVu', 7.4)
        canv.setFillColor(MUTED)
        # running header
        canv.drawString(self.leftMargin, A4[1] - 12 * mm, self._h1[:78])
        canv.setStrokeColor(CODE_BR)
        canv.setLineWidth(0.4)
        canv.line(self.leftMargin, A4[1] - 13.6 * mm,
                  A4[0] - self.rightMargin, A4[1] - 13.6 * mm)
        # footer
        canv.drawRightString(A4[0] - self.rightMargin, 10 * mm, str(canv.getPageNumber()))
        canv.drawString(self.leftMargin, 10 * mm, 'Analytic_t_dep_sim_mem_saver.py')
        canv.restoreState()

    def afterFlowable(self, flowable):
        if not hasattr(flowable, 'style'):
            return
        sn = flowable.style.name
        txt = getattr(flowable, '_raw_text', None)
        if txt is None:
            return
        txt = plain(txt)
        if sn == 'H1':
            self._h1 = txt
            lvl = 0
        elif sn == 'H2':
            lvl = 1
        elif sn == 'H3':
            lvl = 2
        else:
            return
        self._seq += 1
        key = f'sec{self._seq}'
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(txt[:110], key, level=lvl, closed=(lvl > 0))
        self.notify('TOCEntry', (lvl, txt, self.page, key))


# ----------------------------------------------------------------- flowable builders
def _tagged(par, raw):
    par._raw_text = raw
    return par


def build_flowables(content, S):
    out = []
    for kind, payload in content:
        if kind == 'TITLE':
            t, sub = payload
            out += [Spacer(1, 52 * mm),
                    Paragraph(md(t), S['title']),
                    Spacer(1, 4 * mm)]
            for line in sub:
                # An empty Paragraph collapses to zero height; use real space.
                out.append(Spacer(1, 5 * mm) if not line.strip()
                           else Paragraph(md(line), S['subtitle']))
        elif kind == 'H1':
            out.append(CondPageBreak(60 * mm))
            out.append(_tagged(Paragraph(md(payload), S['h1']), payload))
        elif kind == 'H2':
            out.append(CondPageBreak(34 * mm))
            out.append(_tagged(Paragraph(md(payload), S['h2']), payload))
        elif kind == 'H3':
            out.append(CondPageBreak(26 * mm))
            out.append(_tagged(Paragraph(md(payload), S['h3']), payload))
        elif kind == 'P':
            out.append(Paragraph(md(payload), S['body']))
        elif kind == 'CODE':
            # Preformatted honours neither backColor nor borderColor, so box it
            # in a Table. A one-row Table cannot split, so chunk long listings
            # into page-sized rows -- the Table then breaks between them.
            lines = payload.rstrip('\n').split('\n')
            rows = [['\n'.join(lines[i:i + CODE_ROWS])]
                    for i in range(0, len(lines), CODE_ROWS)]
            data = [[Preformatted(r[0], S['code'])] for r in rows]
            t = Table(data, colWidths=[168 * mm], hAlign='LEFT')
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), CODE_BG),
                ('BOX', (0, 0), (-1, -1), 0.6, CODE_BR),
                ('LEFTPADDING', (0, 0), (-1, -1), 7),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 0),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                ('TOPPADDING', (0, 0), (0, 0), 6),
                ('BOTTOMPADDING', (0, -1), (0, -1), 6),
            ]))
            out += [t, Spacer(1, 9)]
        elif kind == 'MATH':
            for i, line in enumerate(payload.split('\n')):
                st = S['math']
                out.append(Paragraph(_esc(line).replace(' ', '&nbsp;'), st))
        elif kind == 'BULLETS':
            for b in payload:
                out.append(Paragraph(md(b), S['bullet'], bulletText='•'))
            out.append(Spacer(1, 4))
        elif kind == 'NUMS':
            for i, b in enumerate(payload, 1):
                out.append(Paragraph(md(b), S['bullet'], bulletText=f'{i}.'))
            out.append(Spacer(1, 4))
        elif kind == 'NOTE':
            out.append(Paragraph(md(payload), S['note']))
        elif kind == 'WARN':
            out.append(Paragraph(md(payload), S['warn']))
        elif kind == 'CAP':
            out.append(Paragraph(md(payload), S['cap']))
        elif kind == 'TABLE':
            hdr, rows, widths, mono_cols = payload
            mono_cols = mono_cols or ()
            data = [[Paragraph(md(h), S['cellh']) for h in hdr]]
            for r in rows:
                data.append([
                    Paragraph(md(c), S['cellm'] if j in mono_cols else S['cell'])
                    for j, c in enumerate(r)])
            total = 168 * mm
            w = [total * x / sum(widths) for x in widths]
            t = Table(data, colWidths=w, repeatRows=1, hAlign='LEFT')
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('GRID', (0, 0), (-1, -1), 0.4, CODE_BR),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, CODE_BG]),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 3.5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
            ]))
            out += [t, Spacer(1, 9)]
        elif kind == 'PAGEBREAK':
            out.append(PageBreak())
        elif kind == 'SPACE':
            out.append(Spacer(1, payload * mm))
        else:
            raise ValueError(f'unknown kind {kind}')
    return out


def render(content, path):
    register_fonts()
    S = build_styles()
    doc = Doc(path)

    toc = TableOfContents()
    toc.levelStyles = S['toc']
    toc.dotsMinLevel = 0

    story = []
    # cover
    head = []
    i = 0
    if content and content[0][0] == 'TITLE':
        head = build_flowables(content[:1], S)
        i = 1
    story += head
    story.append(PageBreak())
    story.append(Paragraph('Contents', S['h1']))
    story.append(toc)
    story.append(reportlab_next_template())   # takes effect at the next break
    story.append(PageBreak())
    story += build_flowables(content[i:], S)

    doc.multiBuild(story)
    return path


def reportlab_next_template():
    from reportlab.platypus import NextPageTemplate
    return NextPageTemplate('body')
