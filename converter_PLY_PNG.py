import os
import open3d as o3d
import numpy as np


def render_isometric_views(pcd, output_dir, base_name, width=512, height=512):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)
    vis.get_render_option().background_color = np.array([1, 1, 1])

    view_control = vis.get_view_control()

    camera_directions = [
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [-1.0, -1.0, 1.0],  # Optional fourth view
    ]

    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    max_extent = np.linalg.norm(extent)

    for i, front_vector in enumerate(camera_directions):
        front = np.array(front_vector)
        front /= np.linalg.norm(front)
        up = np.array([0, 0, 1])
        distance = max_extent * 1.5
        eye_position = center + front * distance

        view_control.set_lookat(center)
        view_control.set_front(front)
        view_control.set_up(up)
        view_control.set_zoom(0.7)

        vis.poll_events()
        vis.update_renderer()

        output_file = os.path.join(output_dir, f"{base_name}_iso_{i}.png")
        vis.capture_screen_image(output_file)
        print(f"Saved image: {output_file}")

    vis.destroy_window()


input_root = r"XXXXXXXXXXXXXX" # Your input path
output_root = r"XXXXXXXXXXXXX" # Your output path

for dirpath, _, filenames in os.walk(input_root):
    for filename in filenames:
        if filename.lower().endswith('.ply'):
            input_path = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(input_path, input_root)
            relative_folder = os.path.dirname(relative_path)
            base_name = os.path.splitext(os.path.basename(filename))[0]
            output_folder = os.path.join(output_root, relative_folder)
            os.makedirs(output_folder, exist_ok=True)

            try:
                point_cloud = o3d.io.read_point_cloud(input_path)
                render_isometric_views(point_cloud, output_folder, base_name)
            except Exception as e:
                print(f"Error processing {input_path}: {e}")
