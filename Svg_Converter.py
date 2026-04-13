import cv2
from functools import lru_cache
import numpy as np
from svgpathtools import Path, Line, wsvg
from math import sin
import os
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import Voronoi, Delaunay, cKDTree
import svgwrite


class SvgConverter:

    def __init__(self, imagePath: str, outPath: str):
        self.imagePath = imagePath
        self.outPath = outPath
        self.image = cv2.imread(self.imagePath)

    def frange(self, start, stop, increment=1.0):
        current = start
        while current < stop:
            yield current
            current += increment

    def addWhiteBorders(self, image, border_size=10):
        bordered = cv2.copyMakeBorder(
            image,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        return bordered

    @staticmethod
    def resize_image(image: np.ndarray, height: int):
        aspect_ratio = float(image.shape[1]) / float(image.shape[0])
        width = int(height * aspect_ratio)
        resized_image = cv2.resize(image, (width, height))
        return resized_image

    @lru_cache
    def get_range_val(self,start, end, increment, idx):
        return list(self.frange(start, end, increment))[::-1][idx]

    def SvgToSin(self, height : int, pixel_width : float ,resolution = 0.7  , max_amplitude : float =2.0 , max_frequency=3  , white :int = 230, White_Removal : bool = False):

        image = cv2.imread(self.imagePath)
        if White_Removal:
            self.image = self.addWhiteBorders(image , 15)
        outpath_white = "TestingImages/ProcessedWhite.svg"
        image = self.resize_image(image , height)  # adjust height
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        WHITE_THRESHOLD = white
        all_lines = []
        white_paths = []

        for row in range(image.shape[0]):
            line_start_height = (row * pixel_width) + (pixel_width / 2)
            current_x = 0
            current_sin_amplitude = 0
            current_sin_frequency = 20
            current_sin_phase = 0

            sin_line = Path()
            white_line = Path()
            in_white = False

            start_point = complex(current_x, line_start_height)

            for col in range(image.shape[1]):
                pixel = image[row, col]

                target_amp = self.get_range_val(0, max_amplitude, max_amplitude / 255, pixel)
                target_freq = self.get_range_val(0, max_frequency, max_frequency / 255, pixel)

                for _ in self.frange(0, pixel_width, resolution):

                    current_sin_amplitude += (target_amp - current_sin_amplitude) * resolution
                    current_sin_frequency += (target_freq - current_sin_frequency) * resolution
                    current_sin_phase += current_sin_frequency * resolution

                    current_y = (current_sin_amplitude * sin(current_sin_phase)) + line_start_height
                    end_point = complex(current_x, current_y)
                    line = Line(start_point, end_point)

                    if pixel >= WHITE_THRESHOLD:

                        if not in_white:

                            if len(sin_line) > 0:
                                all_lines.append(sin_line)
                                sin_line = Path()
                            in_white = True
                        white_line.append(line)
                        sin_line.append(line)
                    else:

                        if in_white:

                            if len(white_line) > 0:
                                white_paths.append(white_line)
                                white_line = Path()
                            in_white = False
                        sin_line.append(line)

                    current_x += resolution
                    start_point = end_point

            if len(sin_line) > 0:
                all_lines.append(sin_line)
            if len(white_line) > 0:
                white_paths.append(white_line)

        wsvg(paths=all_lines, filename=self.outPath)
        wsvg(paths=white_paths, filename=outpath_white)
        svg_white = f"{outpath_white}"
        svg = f"{self.outPath}"

        return svg, svg_white



    def voronoi_tsp_pipeline(self,
        image_path ,
        output_dir="",
        num_points=20000,
        edge_fraction=0.6,
        relax_iters=12,
        relax_alpha=0.15,
        canny_low=50,
        canny_high=150,
        brightness_exp=2.0,
        edge_weight=2.0,
        draw_vor_edges=False,
        make_single_path=True,
        svg_size_scale=1.3,
        rng_seed=42
    ):
        """
        Full edge-aware Voronoi TSP pipeline with:
        - Hybrid edge + brightness sampling
        - Weighted Lloyd relaxation
        - Voronoi construction
        - Clipping of Voronoi segments to image bounds
        - Greedy stitching of segments into a single continuous polyline
        Saves PNG and SVG into output_dir. Returns nothing.
        """
        image_path = self.imagePath
        np.random.seed(rng_seed)
        os.makedirs(output_dir, exist_ok=True)

        # --- Helpers ---
        def load_image(path):
            img = Image.open(path).convert("RGB")
            arr = np.array(img)
            h, w = arr.shape[:2]
            return arr, w, h

        def compute_gray_and_edges(rgb, low, high):
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, low, high)
            return gray, edges

        def sample_edge_points(edges, count):
            coords = np.column_stack(np.where(edges > 0))  # (y,x)
            if len(coords) == 0:
                return np.empty((0, 2), dtype=int)
            count = min(count, len(coords))
            idx = np.random.choice(len(coords), count, replace=False)
            sel = coords[idx]
            return np.stack([sel[:,1], sel[:,0]], axis=1)  # to (x,y)

        def sample_brightness_points(gray, count, width, height, exp=2.0):
            pts = []
            while len(pts) < count:
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                bright = gray[y, x] / 255.0
                prob = (1.0 - bright) ** exp
                if np.random.rand() < prob:
                    pts.append([x, y])
            return np.array(pts, dtype=float)

        def weighted_lloyd(points, gray, edges, iters=10, alpha=0.1, edge_weight=1.0, brightness_exp=2.0):
            h, w = gray.shape
            bweight = (1.0 - (gray.astype(float) / 255.0)) ** brightness_exp
            eweight = np.where(edges > 0, edge_weight, 1.0)
            pix_weight = bweight * eweight

            for _ in range(iters):
                delaunay = Delaunay(points)
                centroids = np.zeros_like(points, dtype=float)
                weights = np.zeros(len(points), dtype=float)

                # stride for speed; adjust to 1 for max accuracy
                for i in range(0, w, 2):
                    for j in range(0, h, 2):
                        weight = pix_weight[j, i]
                        if weight <= 0:
                            continue
                        simplex = delaunay.find_simplex([[i, j]])
                        if simplex >= 0:
                            verts = delaunay.simplices[simplex][0]
                            idx = verts[0]
                            centroids[idx] += np.array([i, j]) * weight
                            weights[idx] += weight

                mask = weights > 0
                new_pts = points.copy()
                new_pts[mask] = points[mask] + alpha * (
                    (centroids[mask] / weights[mask][:, None]) - points[mask]
                )
                points = new_pts
            return points

        # Clip helpers for Voronoi segments
        def clip_point(p, width, height):
            return np.array([np.clip(p[0], 0, width - 1),
                             np.clip(p[1], 0, height - 1)], dtype=float)

        def voronoi_segments_clipped(vor, width, height):
            """
            Build finite Voronoi segments clipped to image bounds.
            Skips ridges with -1 (infinite) and clamps coordinates.
            """
            segments = []
            for ridge in vor.ridge_vertices:
                if -1 in ridge:
                    continue  # skip infinite edges
                v1, v2 = ridge
                p1 = vor.vertices[v1]
                p2 = vor.vertices[v2]
                p1c = clip_point(p1, width, height)
                p2c = clip_point(p2, width, height)
                segments.append((p1c[0], p1c[1], p2c[0], p2c[1]))
            return segments

        def plot_points_and_voronoi(points, vor, rgb, draw_edges=True, fname_png=None, width=None, height=None):
            plt.figure(figsize=(10, 10))
            plt.axis("equal")
            plt.axis("off")
            plt.scatter(points[:,0], points[:,1], s=1, c="black")
            if draw_edges:
                segs = voronoi_segments_clipped(vor, width, height) if (width is not None and height is not None) else []
                for x1, y1, x2, y2 in segs:
                    plt.plot([x1, x2], [y1, y2], color="black", linewidth=0.5)
            if fname_png:
                plt.savefig(fname_png, dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close()

        # Stitch clipped segments into a single continuous polyline (greedy)
        def stitch_segments_to_polyline(segments):
            """
            Greedy stitching: connect segment endpoints by nearest neighbor to form one continuous polyline.
            Not graph-optimal but effective to avoid pen lifts.
            """
            if not segments:
                return np.empty((0,2))

            # Collect endpoints
            endpoints = []
            for x1, y1, x2, y2 in segments:
                endpoints.append([x1, y1])
                endpoints.append([x2, y2])
            endpoints = np.array(endpoints, dtype=float)

            # KDTree for nearest neighbor
            tree = cKDTree(endpoints)
            n = len(endpoints)
            visited = np.zeros(n, dtype=bool)
            path = []

            # start at endpoint closest to the mean for central start
            start_target = endpoints.mean(axis=0)
            _, start_idx = tree.query(start_target)
            current = start_idx
            visited[current] = True
            path.append(endpoints[current])

            for _ in range(n - 1):
                # query k nearest to find the next unused
                dists, idxs = tree.query(endpoints[current], k=min(10, n))
                found = False
                for candidate in np.atleast_1d(idxs):
                    if not visited[candidate]:
                        visited[candidate] = True
                        path.append(endpoints[candidate])
                        current = candidate
                        found = True
                        break
                if not found:
                    # fallback: any remaining nearest
                    unvisited = np.where(~visited)[0]
                    if len(unvisited) == 0:
                        break
                    sub_tree = cKDTree(endpoints[unvisited])
                    _, loc = sub_tree.query(endpoints[current])
                    next_idx = unvisited[loc]
                    visited[next_idx] = True
                    path.append(endpoints[next_idx])
                    current = next_idx

            return np.vstack(path)

        def export_svg_polyline(points, width, height, out_path, scale=1.0):
            dwg = svgwrite.Drawing(out_path, size=(width * scale, height * scale))
            if len(points) == 0:
                dwg.save()
                return

            point_list = [(float(x), float(y)) for x, y in points]
            polyline = dwg.polyline(points=point_list,
                                    stroke="black",
                                    fill="none",
                                    stroke_width=1)
            dwg.add(polyline)
            dwg.save()

        # --- Pipeline ---
        rgb, width, height = load_image(image_path)
        gray, edges = compute_gray_and_edges(rgb, canny_low, canny_high)

        edge_count = int(num_points * edge_fraction)
        bright_count = num_points - edge_count
        edge_pts = sample_edge_points(edges, edge_count).astype(float)
        bright_pts = sample_brightness_points(gray, bright_count, width, height, exp=brightness_exp)
        points = np.vstack([edge_pts, bright_pts])

        points = weighted_lloyd(points, gray, edges, iters=relax_iters,
                                alpha=relax_alpha, edge_weight=edge_weight,
                                brightness_exp=brightness_exp)

        vor = Voronoi(points)

        png_path = os.path.join(output_dir, "voronoi_stipple.png")
        plot_points_and_voronoi(points, vor, rgb, draw_edges=draw_vor_edges,
                                fname_png=png_path, width=width, height=height)

        if make_single_path:
            clipped_segments = voronoi_segments_clipped(vor, width, height)
            path_points = stitch_segments_to_polyline(clipped_segments)
            svg_path = os.path.join(output_dir, "Processed.svg")
            export_svg_polyline(path_points, width, height, svg_path, scale=svg_size_scale)
        else:
            svg_path = os.path.join(output_dir, "stipple_points_path.svg")
            export_svg_polyline(points, width, height, svg_path, scale=svg_size_scale)
