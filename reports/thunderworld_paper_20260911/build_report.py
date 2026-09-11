"""Render the visual English report with embedded vector figures and a PDF export."""
import base64, hashlib, html, json, re, zlib
from pathlib import Path
from markdown_it import MarkdownIt
from playwright.sync_api import sync_playwright
HERE=Path(__file__).resolve().parent
CSS='''
:root{--ink:#173747;--teal:#007f86;--muted:#526776;--line:#d5e1e5}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:#edf2f3;font:16px/1.65 Arial,sans-serif}
aside{position:fixed;inset:0 auto 0 0;width:220px;padding:28px 20px;background:#112c3b;color:#cce1e4;overflow:auto}
aside b{display:block;color:white;font-size:23px;margin-bottom:8px}aside small{display:block;margin-bottom:22px;color:#9ec2ca}aside a{display:block;color:#d4e5e9;text-decoration:none;padding:7px 0;border-bottom:1px solid #294452;font-size:12px;line-height:1.45}
main{margin:0 28px 0 248px;max-width:1400px;background:white;padding:34px 42px 45px}.cover{border-top:5px solid var(--teal);padding-top:18px;margin-bottom:25px}
h1{font-size:38px;margin:0 0 8px;letter-spacing:-1px}h2{font-size:25px;line-height:1.35;color:var(--teal);border-bottom:2px solid var(--teal);padding-bottom:10px;margin:0 0 16px}p{margin:12px 0}
a{color:var(--teal);overflow-wrap:anywhere}section{margin-bottom:52px;scroll-margin-top:20px}img{display:block;width:100%;height:auto;margin:18px auto}code{font-size:.9em}strong{color:var(--ink)}.actions{font-size:13px}.actions a{margin-right:20px}footer{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:14px}
.figure-wrap{overflow:auto}.figure-wrap a{display:block}.figure-wrap img{cursor:zoom-in}.figure-wrap:focus-within{outline:2px solid var(--teal)}
@media(max-width:1000px){aside{display:none}main{margin:0;padding:22px}h1{font-size:30px}h2{font-size:22px}.figure-wrap img{min-width:850px}.figure-wrap{border:1px solid var(--line)}.figure-wrap::after{content:'Scroll to inspect the figure; select it to open the full-size image.';display:block;font-size:12px;padding:8px;color:var(--muted)}}
@page{size:A4 landscape;margin:11mm 13mm 13mm}
@media print{
body{background:white;font-size:10pt;line-height:1.4;-webkit-print-color-adjust:exact;print-color-adjust:exact}aside,.actions,footer{display:none}main{padding:0;margin:0;max-width:none}.cover{border-top:3px solid var(--teal);padding-top:3mm;margin-bottom:3mm}h1{font-size:23pt;margin:0 0 1mm}.cover p{font-size:9pt;margin:0}h2{font-size:17pt;line-height:1.25;margin:0 0 3mm;padding-bottom:2mm}p{margin:2mm 0 3mm;orphans:3;widows:3}section{margin:0;break-before:page}section.summary{break-before:auto}section.summary h2{font-size:12pt;border:0;padding:0;margin:2mm 0}section.summary p{font-size:9pt}a{text-decoration:none}.figure-wrap{overflow:visible;border:0;break-inside:avoid;margin:0}.figure-wrap::after{display:none}.figure-wrap img{min-width:0;width:100%;height:125mm;object-fit:contain;margin:1mm auto 2mm}section.summary .figure-wrap img{height:94mm}.figure-wrap a{color:inherit}section p{font-size:10pt}section.summary .figure-wrap{margin:0}
}
'''
def main():
 text=(HERE/'REPORT.md').read_text();md=MarkdownIt('commonmark',{'html':True}).enable('table');tokens=md.parse(text);toc=[]
 for i,t in enumerate(tokens):
  if t.type=='heading_open' and t.tag=='h2':
   anchor=f'section-{len(toc)}';t.attrSet('id',anchor);toc.append((anchor,tokens[i+1].content))
 body=md.renderer.render(tokens,md.options,{})
 def embed(m):
  path=HERE/m.group(1);assert path.is_file(),path
  mime='image/svg+xml' if path.suffix=='.svg' else 'image/png'
  return 'src="data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()+'"'
 # Linked standalone assets allow inspection at narrow screen widths.
 body=re.sub(r'<p>(<img src="([^"]+)"[^>]*>)</p>',lambda m:f'<div class="figure-wrap"><a href="{m.group(2)}" target="_blank" aria-label="Open full-size figure">{m.group(1)}</a></div>',body)
 body=re.sub(r'src="([^"#]+\.(?:svg|png))"',embed,body)
 parts=re.split(r'(?=<h2 )',body);front=parts.pop(0)
 sections=''.join(f'<section class="{"summary" if i==0 else "visual"}">{part}</section>' for i,part in enumerate(parts))
 nav=''.join(f'<a href="#{anchor}">{html.escape(label)}</a>' for anchor,label in toc)
 doc=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ThunderWorld · Visual Research Report</title><style>{CSS}</style></head><body><aside><b>ThunderWorld</b><small>Visual research report<br>2026 · 09 · 11</small>{nav}</aside><main><div class="cover">{front}<div class="actions"><a href="REPORT.pdf">Visual PDF</a><a href="REPORT.md">Markdown</a><a href="REPORT_DETAILED.pdf">Detailed appendix</a></div></div>{sections}<footer>ThunderWorld · Archived experimental evidence · AI-generated overview + reproducible vector diagrams and charts.</footer></main></body></html>'''
 (HERE/'REPORT.html').write_text(doc);qa=HERE/'qa';qa.mkdir(exist_ok=True)
 errors=[]
 with sync_playwright() as p:
  browser=p.chromium.launch(headless=True,args=['--disable-gpu']);page=browser.new_page(viewport={'width':1600,'height':1050},device_scale_factor=1)
  page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto((HERE/'REPORT.html').as_uri(),wait_until='load');page.evaluate('document.fonts.ready');page.wait_for_function('Array.from(document.images).every(x=>x.complete && x.naturalWidth>0)')
  assert page.locator('img').count()==14
  page.screenshot(path=str(qa/'cover.png'))
  for i in range(len(toc)):
   page.locator('section').nth(i).screenshot(path=str(qa/f'visual_section_{i:02d}.png'))
  page.emulate_media(media='print')
  # Print layout geometry predicts overflow; PDF page count validates the actual pagination.
  layout=page.locator('section').evaluate_all('(xs)=>xs.map(x=>({height:x.getBoundingClientRect().height,title:x.querySelector("h2").textContent}))')
  page.pdf(path=str(HERE/'REPORT.pdf'),prefer_css_page_size=True,print_background=True,display_header_footer=True,header_template='<span></span>',footer_template='<div style="font-size:8px;color:#617480;width:100%;text-align:center">ThunderWorld · Visual Research Report &nbsp; | &nbsp; <span class="pageNumber"></span> / <span class="totalPages"></span></div>')
  page.emulate_media(media='screen');page.set_viewport_size({'width':390,'height':844});page.goto((HERE/'REPORT.html').as_uri(),wait_until='load');page.evaluate('document.fonts.ready')
  page.screenshot(path=str(qa/'mobile.png'))
  overflow=page.evaluate('document.documentElement.scrollWidth > innerWidth+1');assert not overflow,'Mobile page overflow'
  browser.close()
 pdf=(HERE/'REPORT.pdf').read_bytes();objects={int(m.group(1)):m.group(2) for m in re.finditer(rb'(\d+) 0 obj\s*(.*?)\s*endobj',pdf,re.S)};pages=[]
 for number,obj in objects.items():
  if not re.search(rb'/Type\s*/Page\b',obj):continue
  dims=list(map(float,re.search(rb'/MediaBox\s*\[([^]]+)\]',obj).group(1).split()))
  ref=int(re.search(rb'/Contents\s+(\d+) 0 R',obj).group(1));content=objects[ref];data=re.search(rb'stream\r?\n(.*?)\r?\nendstream',content,re.S).group(1)
  if b'/FlateDecode' in content:data=zlib.decompress(data)
  assert len(data)>500
  pages.append(dict(width=dims[2],height=dims[3],content_bytes=len(data)))
 check={'passed':not errors,'figures':14,'sections':len(toc),'pdf_pages':len(pages),'all_landscape':all(x['width']>x['height'] for x in pages),'page_errors':errors,'mobile_page_overflow':overflow,'print_section_geometry':layout,'image_sha256':hashlib.sha256((HERE/'assets/thunderworld_architecture.png').read_bytes()).hexdigest()}
 (qa/'render_validation.json').write_text(json.dumps(check,indent=2)+'\n');(qa/'pdf_structure_validation.json').write_text(json.dumps({'passed':True,'pages':pages},indent=2)+'\n')
 print(json.dumps(check,indent=2));assert not errors
 assert len(pages)==len(toc),(len(pages),len(toc))
if __name__=='__main__':main()
