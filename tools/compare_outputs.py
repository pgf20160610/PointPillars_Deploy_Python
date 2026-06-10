#!/usr/bin/env python3
import argparse, json, math
p=argparse.ArgumentParser(description='Minimal JSON box comparison helper')
p.add_argument('--ref', required=True); p.add_argument('--test', required=True); p.add_argument('--tolerance', type=float, default=1e-3)
a=p.parse_args()
ref=json.load(open(a.ref)); test=json.load(open(a.test))
print(f'ref_boxes={len(ref)} test_boxes={len(test)} tolerance={a.tolerance}')
if len(ref)!=len(test): raise SystemExit(1)
keys=['x','y','z','dx','dy','dz','yaw','score']
for i,(r,t) in enumerate(zip(ref,test)):
    for k in keys:
        if abs(float(r[k])-float(t[k]))>a.tolerance:
            raise SystemExit(f'mismatch box {i} key {k}: {r[k]} vs {t[k]}')
print('OK')
