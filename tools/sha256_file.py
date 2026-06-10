#!/usr/bin/env python3
import argparse, hashlib
from pathlib import Path
p=argparse.ArgumentParser(description='Compute sha256 for model files')
p.add_argument('files', nargs='+')
a=p.parse_args()
for f in a.files:
    path=Path(f); h=hashlib.sha256()
    with path.open('rb') as fp:
        for chunk in iter(lambda: fp.read(1024*1024), b''):
            h.update(chunk)
    print(f'{h.hexdigest()}  {path}')
