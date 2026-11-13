"""Create a deterministic source manifest and curated ZIP; excludes build caches."""
from __future__ import annotations
import hashlib
import json
import platform
import sys
import zipfile
from pathlib import Path
import numpy,scipy,pandas,matplotlib,pytest
root=Path(__file__).resolve().parents[1]
output=root.parent/'IGRM_v17_Adaptive_Research_Package.zip'
paths=[]
for path in sorted(root.rglob('*')):
    if not path.is_file():continue
    parts=path.relative_to(root).parts
    if any(part in ('_build','__pycache__','.pytest_cache') for part in parts):continue
    if path.name in ('CHECKSUMS_SHA256_V17.txt','MANIFEST_V17.json','package_v17.py'):
        continue
    if path.name in ('IGRM_manuscript_v15.pdf','IGRM_manuscript_v16.pdf'):
        continue
    if path.suffix in ('.pyc','.pyo'):
        continue
    # Omit duplicated historical exports but keep the historical source, math and data.
    paths.append(path)
manifest={
    'status':'Research draft; independent expert peer review pending',
    'date':'2026-11-12',
    'manuscript':'IGRM_manuscript_v17.pdf',
    'canonical_v10_sha256':'38ea2b41b12986ac200828a72a53b0afd7d5732ff921f3ed8b0e2bf89ed31c43',
    'v17_experiment_sha256':hashlib.sha256((root/'data/adaptive_overlap_v17.csv').read_bytes()).hexdigest(),
    'v17_rows':7000,
    'v17_unique_streams':1000,
    'v17_tests_passed':65,
    'original_v10_tests_passed':81,
    'sha256_identical_full_repetition':True,
    'environment':{'python':sys.version.split()[0],'os':platform.platform(),
       'numpy':numpy.__version__,'scipy':scipy.__version__,'pandas':pandas.__version__,
       'matplotlib':matplotlib.__version__,'pytest':pytest.__version__},
    'files':[
        {'relative_path':str(p.relative_to(root)), 'size_bytes':p.stat().st_size,
         'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
         for p in paths],
}
manifest_path=root/'MANIFEST_V17.json'
manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
paths.append(manifest_path)
checksum_path=root/'CHECKSUMS_SHA256_V17.txt'
checksum_path.write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}\n'
                                 for p in sorted(paths)),encoding='utf-8')
paths.append(checksum_path)
with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
    for path in sorted(paths):
        z.write(path,'IGRM_Manuscript_v17/'+str(path.relative_to(root)))
print('ZIP',output,'files',len(paths),'bytes',output.stat().st_size,'sha256',hashlib.sha256(output.read_bytes()).hexdigest())
print('main_csv_sha256',manifest['v17_experiment_sha256'])
