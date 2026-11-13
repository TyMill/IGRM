from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
files=[]
for path in sorted(root.rglob('*')):
    if not path.is_file() or '_build' in path.parts or '__pycache__' in path.parts or path.name=='FILE_MANIFEST_v13.json':
        continue
    files.append({'path':str(path.relative_to(root)), 'bytes':path.stat().st_size, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(root/'FILE_MANIFEST_v13.json').write_text(json.dumps({'name':'IGRM Manuscript v1.3','created':'2026-11-12','files':files},indent=2))
