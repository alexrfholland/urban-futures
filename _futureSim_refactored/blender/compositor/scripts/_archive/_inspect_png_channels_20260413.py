"""Sample an RGBA PNG, report per-channel stats over opaque pixels."""
import sys
from pathlib import Path
import bpy

png = Path(sys.argv[-1])
img = bpy.data.images.load(str(png), check_existing=False)
w, h = img.size[0], img.size[1]
px = list(img.pixels)

r, g, b, a = [], [], [], []
for i in range(0, len(px), 4):
    if px[i + 3] > 0.5:
        r.append(px[i + 0]); g.append(px[i + 1]); b.append(px[i + 2]); a.append(px[i + 3])

def stats(label, v):
    if not v:
        print(f"{label}: empty"); return
    mn = min(v); mx = max(v); mean = sum(v) / len(v)
    var = sum((x - mean) ** 2 for x in v) / len(v)
    print(f"{label}: min={mn:.4f} max={mx:.4f} mean={mean:.4f} stdev={var**0.5:.4f}")

print(f"file: {png.name}  size: {w}x{h}  opaque: {len(r)}")
stats("R", r); stats("G", g); stats("B", b); stats("A", a)

# How many opaque pixels have R==0 exactly (classic clipped-negative signature)?
zero_r = sum(1 for v in r if v < 0.002)
zero_g = sum(1 for v in g if v < 0.002)
zero_b = sum(1 for v in b if v < 0.002)
print(f"clipped-to-zero counts:  R={zero_r}  G={zero_g}  B={zero_b}  (of {len(r)})")
