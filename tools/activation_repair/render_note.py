"""Render the English study as a self-contained local HTML report."""
from pathlib import Path
import base64,re,json
from markdown_it import MarkdownIt
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[2];src=ROOT/'docs/ACTIVATION_PAIR_AND_CALIBRATION_20260911.md';out=src.with_suffix('.html')
body=MarkdownIt('commonmark').enable('table').render(src.read_text())
def embed(m):
 p=(src.parent/m[1]).resolve()
 return 'src="data:image/svg+xml;base64,'+base64.b64encode(p.read_bytes()).decode()+'"'
body=re.sub(r'src="([^\"]+\.svg)"',embed,body)
css='''body{margin:0;background:#edf2f3;color:#173747;font:16px/1.65 Arial,sans-serif}main{max-width:1100px;margin:28px auto;background:white;padding:38px 48px}h1{font-size:32px;line-height:1.25;border-top:5px solid #007f86;padding-top:18px}h2{color:#007f86;margin-top:45px;border-bottom:1px solid #d5e1e5}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow:auto}td,th{padding:9px 12px;text-align:left;border-bottom:1px solid #d5e1e5}th{background:#e5f2f1}a{color:#007f86}code{overflow-wrap:anywhere}p{margin:16px 0}@media(max-width:700px){main{margin:0;padding:18px}body{font-size:15px}h1{font-size:27px}}@page{size:A4;margin:15mm}@media print{body{background:white;font-size:10pt}main{margin:0;padding:0;max-width:none}h1{font-size:23pt}h2{font-size:16pt;break-after:avoid}img{break-inside:avoid}table{font-size:9pt}tr{break-inside:avoid}a{color:inherit;text-decoration:none}}'''
out.write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ThunderWorld — Activation Pairing and Calibration</title><style>{css}</style><main>{body}</main></html>')
qa=ROOT/'results/activation_repair_20260911/render';qa.mkdir(exist_ok=True)
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--disable-gpu']);page=browser.new_page(viewport={'width':1400,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(out.as_uri());page.wait_for_function('Array.from(document.images).every(x=>x.complete&&x.naturalWidth>0)');assert page.locator('img').count()==2
 page.screenshot(path=str(qa/'desktop.png'));page.pdf(path=str(src.with_suffix('.pdf')),prefer_css_page_size=True,print_background=True)
 page.set_viewport_size({'width':390,'height':844});page.screenshot(path=str(qa/'mobile.png'));overflow=page.evaluate('document.documentElement.scrollWidth>innerWidth+1');assert not overflow;assert not errors
 (qa/'checks.json').write_text(json.dumps({'images':2,'page_errors':errors,'mobile_overflow':overflow,'html':str(out)},indent=2)+'\n');browser.close()
print(out)
