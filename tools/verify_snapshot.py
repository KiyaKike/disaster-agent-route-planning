"""Verify only the files included in this GitHub research snapshot."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'snapshot_manifest.json').read_text())
for row in manifest['files']:
    p=root/row['path']
    assert p.is_file(),row['path']
    assert p.stat().st_size==row['bytes'],row['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],row['path']
print(f"Verified {len(manifest['files'])} exported files;original large data/caches are intentionally not included.")
