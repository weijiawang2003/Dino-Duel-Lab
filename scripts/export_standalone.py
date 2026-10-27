#!/usr/bin/env python3
"""Export a report with its CSV/JSON/replay links embedded, without model binaries."""
from __future__ import annotations
import argparse,base64,html,json,mimetypes,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def export(source:Path,destination:Path):
    source=source.resolve();root=source.parent
    content=source.read_text(encoding='utf-8');embedded={}
    def replace(match):
        before,quote,link,after=match.groups()
        if link.startswith(('http:','https:','#','data:')):return match.group(0)
        path=(root/html.unescape(link)).resolve()
        if not path.is_relative_to(root):raise ValueError('Report link escapes report root: '+link)
        if path.suffix not in {'.html','.json','.csv'}:
            return f'<span{before}>{after}</span>'
        if not path.is_file():raise FileNotFoundError('Broken report link: '+str(path))
        key=str(path.relative_to(root))
        embedded[key]={'base64':base64.b64encode(path.read_bytes()).decode('ascii'),
                       'mime':mimetypes.guess_type(path.name)[0] or 'application/octet-stream',
                       'name':path.name,'open':path.suffix=='.html'}
        return '<a'+before+' href="#" data-embedded="'+html.escape(key,quote=True)+'">'+after+'</a>'
    content=re.sub(r'<a([^>]*?)\s+href=([\"\x27])(.*?)\2\s*>(.*?)</a>',replace,content,flags=re.S)
    # No arbitrary executable data; payload is base64 of known report attachments.
    payload=json.dumps(embedded,separators=(',',':')).replace('<','\\u003c')
    js='''<script id="embedded-files" type="application/json">__DATA__</script><script>
const embeddedFiles=JSON.parse(document.getElementById('embedded-files').textContent);
document.addEventListener('click',event=>{
 const anchor=event.target.closest('a[data-embedded]');if(!anchor)return;
 event.preventDefault();const f=embeddedFiles[anchor.dataset.embedded];
 const bytes=Uint8Array.from(atob(f.base64),c=>c.charCodeAt(0));
 const url=URL.createObjectURL(new Blob([bytes],{type:f.mime}));
 if(f.open){window.open(url,'_blank','noopener');}
 else{const a=document.createElement('a');a.href=url;a.download=f.name;document.body.appendChild(a);a.click();a.remove();}
 setTimeout(()=>URL.revokeObjectURL(url),120000);
});</script>'''.replace('__DATA__',payload)
    content=content.replace('离线附件需与项目目录一起保留。','此独立页面已内嵌CSV、JSON与回放；权重和完整代码见项目压缩包。')
    content=content.replace('</html>',js+'</html>')
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(content,encoding='utf-8')
    return {'output':str(destination),'embedded_files':len(embedded),'bytes':destination.stat().st_size}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'elimination/report.html')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(export(a.source,a.out),indent=2))
if __name__=='__main__':main()
