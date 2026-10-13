import os
import numpy as np
import open3d as o3d


def read_off_with_faces(filename):
    with open(filename, 'r') as f:
        if f.readline().strip() != 'OFF':
            raise ValueError('Invalid OFF file format.')
        n_verts, n_faces, _ = map(int, f.readline().strip().split())
        verts = np.array([list(map(float, f.readline().strip().split())) for _ in range(n_verts)])
        faces = []
        for _ in range(n_faces):
            parts = list(map(int, f.readline().strip().split()))
            if parts[0] == 3:
                faces.append(parts[1:])
        return verts, np.array(faces)


def triangle_area(v0, v1, v2):
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))


def sample_triangle_surface(v0, v1, v2, n_samples):
    u = np.random.rand(n_samples)
    v = np.random.rand(n_samples)
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - u - v
    samples = np.outer(w, v0) + np.outer(u, v1) + np.outer(v, v2)
    return samples


def generate_fixed_pointcloud(vertices, faces, total_points=3000):
    areas = np.array([triangle_area(*vertices[face]) for face in faces])
    area_sum = areas.sum()
    points = []

    points_per_face = np.floor((areas / area_sum) * total_points).astype(int)
    residual = total_points - points_per_face.sum()
    if residual > 0:
        largest_faces = np.argsort(-areas)
        for i in range(residual):
            points_per_face[largest_faces[i]] += 1

    for face, n_points in zip(faces, points_per_face):
        if n_points > 0:
            v0, v1, v2 = vertices[face]
            sampled = sample_triangle_surface(v0, v1, v2, n_points)
            points.append(sampled)

    return np.vstack(points)


def points_to_ply(output_path, coordinates, rgb=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coordinates)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    o3d.io.write_point_cloud(output_path, pcd)


input_root = r"XXXXXXXXXXXXXXXXXXXX"  #Your input path
output_root = r"XXXXXXXXXXXXx" # Your output path

for dirpath, _, filenames in os.walk(input_root):
    for filename in filenames:
        if filename.endswith('.off'):
            in_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(in_path, input_root)
            out_path = os.path.join(output_root, rel_path)
            out_path = os.path.splitext(out_path)[0] + ".ply"

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            try:
                verts, faces = read_off_with_faces(in_path)
                dense_points = generate_fixed_pointcloud(verts, faces, total_points=3000)
                color = np.ones_like(dense_points) * [0.2, 0.6, 1.0]
                points_to_ply(out_path, dense_points, rgb=color)
                print(f"{rel_path} → saved as point cloud ({dense_points.shape[0]} points)")
            except Exception as e:
                print(f"Error processing {in_path}: {e}")
