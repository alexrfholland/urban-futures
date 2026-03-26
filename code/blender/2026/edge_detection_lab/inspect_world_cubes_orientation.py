import bpy
import math
import numpy as np


TARGET_OBJECTS = (
    "city_highResRoad.001",
    "city_buildings.001",
)


def dominant_xy_angle(obj_name: str) -> None:
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != "MESH":
        print(f"SKIP {obj_name}: missing or not mesh")
        return

    mesh = obj.data
    bins = {}
    total_edges = 0
    for edge in mesh.edges:
        v1 = mesh.vertices[edge.vertices[0]].co
        v2 = mesh.vertices[edge.vertices[1]].co
        dx = v2.x - v1.x
        dy = v2.y - v1.y
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        angle = angle % 90.0
        length = math.hypot(dx, dy)
        bucket = round(angle, 1)
        bins[bucket] = bins.get(bucket, 0.0) + length
        total_edges += 1

    top = sorted(bins.items(), key=lambda item: item[1], reverse=True)[:12]
    print(f"OBJECT {obj_name}")
    print(f"EDGE_COUNT {total_edges}")
    print(f"TOP_ORIENTATIONS {top}")

    if len(mesh.vertices) >= 2:
        pts = np.array([(v.co.x, v.co.y) for v in mesh.vertices], dtype=float)
        center = pts.mean(axis=0)
        pts_centered = pts - center
        cov = np.cov(pts_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, np.argmax(eigvals)]
        pca_angle = math.degrees(math.atan2(axis[1], axis[0])) % 180.0
        print(f"PCA_ANGLE_DEG {round(pca_angle, 3)}")


def main() -> None:
    for obj_name in TARGET_OBJECTS:
        dominant_xy_angle(obj_name)


if __name__ == "__main__":
    main()
