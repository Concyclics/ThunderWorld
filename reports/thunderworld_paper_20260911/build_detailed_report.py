"""Render the source-backed report locally; embed its AI figure for offline use."""
import base64
import hashlib
import html
import json
from pathlib import Path
import re
import zlib
from markdown_it import MarkdownIt
from playwright.sync_api import sync_playwright

HERE=Path(__file__).resolve().parent
CSS='''
:root{--ink:#172e40;--teal:#006f7c;--muted:#526776;--line:#d5e1e5}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:#edf2f3;font-family:Arial,"Droid Sans Fallback",sans-serif;font-size:16px;line-height:1.85}
aside{position:fixed;inset:0 auto 0 0;width:235px;padding:35px 22px;background:#112c3b;color:#cce1e4;overflow:auto}
aside b{display:block;color:white;font-size:24px;margin-bottom:8px}aside small{display:block;margin-bottom:30px;color:#8fb5bf;line-height:1.6}
aside a{display:block;color:#d4e5e9;text-decoration:none;padding:8px 0;border-bottom:1px solid #294452;font-size:13px;line-height:1.6}
main{max-width:1260px;margin:0 30px 0 265px;background:white;padding:50px 60px 70px;box-shadow:0 0 30px #163e4910}
.eyebrow{font-size:12px;letter-spacing:2px;color:var(--teal);font-weight:bold}.cover{border-top:5px solid var(--teal);padding-top:24px;margin-bottom:40px}
h1{font-size:35px;line-height:1.45;letter-spacing:-.8px;margin:12px 0 22px}h2{font-size:25px;line-height:1.5;margin:0 0 20px;padding-bottom:12px;border-bottom:2px solid var(--teal);color:var(--teal)}
h3{font-size:19px;line-height:1.6;margin:26px 0 12px;color:#183f56}section{margin:0 0 45px;scroll-margin-top:24px}p{margin:12px 0 16px}strong{color:#123c4d}
a{color:var(--teal);text-underline-offset:3px;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.7;margin:20px 0 24px;table-layout:auto}
th{background:#e8f3f4;color:#064d58;font-weight:700;text-align:left}td,th{padding:10px 11px;border:1px solid var(--line);vertical-align:top;overflow-wrap:anywhere}tbody tr:nth-child(even){background:#f8fafb}
blockquote{margin:22px 0;padding:12px 23px;border-left:4px solid #c7862d;background:#fff7e9;font-size:18px;line-height:1.8}blockquote p{margin:5px 0}
pre{background:#f0f5f6;border:1px solid var(--line);border-radius:5px;padding:18px;overflow:auto;white-space:pre-wrap;font-size:13px;line-height:1.7}
code{font-family:"Liberation Mono",monospace;font-size:.88em;overflow-wrap:anywhere}p code,td code{background:#edf3f4;padding:1px 4px}
img{display:block;width:100%;height:auto;margin:22px auto;border:1px solid var(--line)}.figure-note{font-size:13px;color:var(--muted)}.actions{font-size:13px;margin-top:20px}.actions a{margin-right:18px}
footer{border-top:1px solid var(--line);padding-top:18px;font-size:12px;color:var(--muted)}
@media(max-width:1000px){aside{display:none}main{margin:0;max-width:none;padding:25px}h1{font-size:28px}table{font-size:12px}td,th{padding:7px}}
@page{size:A4;margin:15mm 13mm 17mm}
@page architecture{size:A4 landscape;margin:13mm 14mm}
@media print{
body{background:white;font-size:9.5pt;line-height:1.55;-webkit-print-color-adjust:exact;print-color-adjust:exact}
aside,.actions{display:none}main{margin:0;padding:0;max-width:none;box-shadow:none}.cover{margin-bottom:9mm;padding-top:5mm}h1{font-size:24pt;line-height:1.35;margin:4mm 0}
h2{font-size:16pt;padding-bottom:2mm;margin-bottom:4mm;break-after:avoid}h3{font-size:11.5pt;margin:5mm 0 2mm;break-after:avoid}
p{margin:2mm 0 3mm;orphans:3;widows:3}section{margin:0 0 7mm;break-before:page}section.summary{break-before:auto}
table{font-size:8.2pt;line-height:1.45;margin:3mm 0 4mm}td,th{padding:2mm 2.3mm}tr{break-inside:avoid}thead{display:table-header-group}
pre{font-size:8.5pt;padding:3mm;margin:3mm 0;break-inside:avoid}blockquote{font-size:11pt;margin:4mm 0;padding:3mm 5mm;break-inside:avoid}a{text-decoration:none}
img{break-inside:avoid;margin:3mm auto}section.architecture{page:architecture}section.architecture img{height:118mm;width:100%;object-fit:contain;border:0}
section.architecture p{font-size:8.5pt;line-height:1.45}.eyebrow{font-size:8pt}footer{font-size:8pt}
}
'''


def main():
    md=MarkdownIt('commonmark',{'html':True}).enable('table')
    text=(HERE/'REPORT_DETAILED.md').read_text();tokens=md.parse(text);toc=[]
    for i,t in enumerate(tokens):
        if t.type=='heading_open' and t.tag=='h2':
            anchor=f'section-{len(toc)}';t.attrSet('id',anchor);toc.append((anchor,tokens[i+1].content))
    body=md.renderer.render(tokens,md.options,{})
    # Only embed local raster files. This preserves the generated image bytes.
    def embed(m):
        source=(HERE/m.group(1)).resolve();assert source.is_file(),source
        return 'src="data:image/png;base64,'+base64.b64encode(source.read_bytes()).decode()+'"'
    body=re.sub(r'src="([^"#]+\.png)"',embed,body)
    sections=re.split(r'(?=<h2 )',body);front=sections.pop(0)
    wrapped=[]
    for i,part in enumerate(sections):
        klass='summary' if i==0 else 'architecture' if i==3 else ''
        wrapped.append(f'<section class="{klass}">{part}</section>')
    nav=''.join(f'<a href="#{anchor}">{html.escape(label)}</a>' for anchor,label in toc)
    doc=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ThunderWorld · Research Narrative and Implementation Evidence</title><style>{CSS}</style></head><body>
<aside><b>ThunderWorld</b><small>Research Narrative and<br>Implementation Evidence<br>2026 · 09 · 11</small>{nav}</aside>
<main><div class="cover"><div class="eyebrow">RESEARCH NARRATIVE / METHODS / EVIDENCE</div>{front}<div class="actions"><a href="REPORT_DETAILED.pdf">PDF edition</a><a href="REPORT_DETAILED.md">Markdown source</a><a href="assets/thunderworld_architecture.png">Architecture figure</a></div></div>
{''.join(wrapped)}<footer>ThunderWorld · 2026-09-11 · Figure generated with the built-in image_gen tool; experimental results sourced from repository archives.</footer></main></body></html>'''
    (HERE/'REPORT_DETAILED.html').write_text(doc)
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--disable-gpu'])
        page=browser.new_page(viewport={'width':1440,'height':1050},device_scale_factor=1)
        page.goto((HERE/'REPORT_DETAILED.html').as_uri(),wait_until='load')
        page.evaluate('document.fonts.ready');page.wait_for_function('Array.from(document.images).every(x=>x.complete && x.naturalWidth>0)')
        broken=page.evaluate('Array.from(document.querySelectorAll("table,pre,img")).filter(x=>x.getBoundingClientRect().right>innerWidth+1).map(x=>x.tagName)')
        assert not broken,broken
        (HERE/'qa/detailed').mkdir(parents=True,exist_ok=True)
        page.screenshot(path=str(HERE/'qa/cover.png'))
        page.locator('section.architecture').screenshot(path=str(HERE/'qa/architecture_section.png'))
        page.locator('#section-7').scroll_into_view_if_needed();page.screenshot(path=str(HERE/'qa/results_section.png'))
        page.pdf(path=str(HERE/'REPORT_DETAILED.pdf'),print_background=True,prefer_css_page_size=True,display_header_footer=True,
            header_template='<span></span>',footer_template='<div style="font-size:8px;color:#617480;width:100%;text-align:center">ThunderWorld · 2026-09-11 &nbsp; | &nbsp; <span class="pageNumber"></span> / <span class="totalPages"></span></div>')
        browser.close()
    pdf=(HERE/'REPORT_DETAILED.pdf').read_bytes();assert pdf.startswith(b'%PDF-')
    # Structural checks for this Chromium-generated PDF, not a general parser.
    objects={int(m.group(1)):m.group(2) for m in re.finditer(rb'(\d+) 0 obj\s*(.*?)\s*endobj',pdf,re.S)}
    pages=[]
    for number,obj in objects.items():
        if not re.search(rb'/Type\s*/Page\b',obj):continue
        dims=list(map(float,re.search(rb'/MediaBox\s*\[([^]]+)\]',obj).group(1).split()))
        ref=int(re.search(rb'/Contents\s+(\d+) 0 R',obj).group(1));content=objects[ref]
        data=re.search(rb'stream\r?\n(.*?)\r?\nendstream',content,re.S).group(1)
        if b'/FlateDecode' in content:data=zlib.decompress(data)
        assert len(data)>500,(number,len(data))
        pages.append(dict(object=number,width=dims[2],height=dims[3],content_bytes=len(data)))
    fonts=sorted({x.decode() for x in re.findall(rb'/BaseFont\s*/([^\s/]+)',pdf)})
    assert sum(x['width']>x['height'] for x in pages)==1
    assert fonts, 'PDF must contain fonts'
    (HERE/'qa/pdf_structure_validation.json').write_text(json.dumps(dict(passed=True,pages=pages,embedded_font_names=fonts),indent=2)+'\n')
    check=dict(passed=True,markdown_chars=len(text),html_images_embedded=True,broken_images=0,horizontal_overflow=broken,
               sections=len(toc),pdf_pages=len(re.findall(rb'/Type\s*/Page\b',pdf)),pdf_bytes=len(pdf),
               image_sha256=hashlib.sha256((HERE/'assets/thunderworld_architecture.png').read_bytes()).hexdigest())
    (HERE/'qa/render_validation.json').write_text(json.dumps(check,indent=2)+'\n');print(json.dumps(check,indent=2))


if __name__=='__main__':main()
