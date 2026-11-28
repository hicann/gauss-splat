# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ast
import json
import os
import struct
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image

try:
    import pycolmap
    PYCOLMAP_SKIP_REASON = ""
except (ImportError, RuntimeError) as exc:
    pycolmap = None
    PYCOLMAP_SKIP_REASON = f"Official pycolmap backend unavailable: {exc.__cause__ or exc}"


ROOT = Path(__file__).resolve().parents[2]


class TensorStub:
    def __init__(self, array):
        self.array = array

    def float(self):
        return self

    def bool(self):
        return self


def load_dataset(image):
    path = ROOT / "examples" / "datasets" / "colmap.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    dataset_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Dataset")
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "Optional": Optional,
        "Parser": object,
        "np": np,
        "imageio": types.SimpleNamespace(imread=lambda _: image.copy()),
        "torch": types.SimpleNamespace(from_numpy=TensorStub),
    }
    exec(compile(ast.Module(body=[dataset_node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Dataset"]


def load_parser(manager, real_paths=False):
    path = ROOT / "examples" / "datasets" / "colmap.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parser_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Parser")
    path_stub = types.SimpleNamespace(join=lambda *parts: "/".join(parts), exists=lambda _: True)
    namespace = {
        "np": np,
        "os": os if real_paths else types.SimpleNamespace(path=path_stub),
        "json": json,
        "pycolmap": types.SimpleNamespace(Reconstruction=lambda _: manager),
        "_get_rel_paths": lambda _: ["a.png", "b.png"],
        "imageio": types.SimpleNamespace(imread=lambda _: np.zeros((6, 8, 3), dtype=np.uint8)),
    }
    exec(compile(ast.Module(body=[parser_node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Parser"]


def parser_stub(mask):
    return types.SimpleNamespace(
        image_names=["image.png"], image_paths=["image.png"], camera_ids=[0], test_every=1,
        Ks_dict={0: np.array([[2., 0., 1.], [0., 2., 1.], [0., 0., 1.]])},
        params_dict={0: np.empty(0)}, camtoworlds=np.eye(4)[None], mask_dict={0: mask},
        point_indices={}, points=np.empty((0, 3), dtype=np.float32),
    )


class ColmapDatasetTest(unittest.TestCase):
    @unittest.skipIf(pycolmap is None, PYCOLMAP_SKIP_REASON)
    def test_official_pycolmap_text_and_binary_models(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        tmp = temp_dir.name
        scene = Path(tmp)
        sparse = scene / "sparse/0"
        sparse.mkdir(parents=True)
        (scene / "images").mkdir()
        (sparse / "cameras.txt").write_text("7 PINHOLE 8 6 4 5 4 3\n")
        (sparse / "images.txt").write_text(
            "9 1 0 0 0 2 0 0 7 b.png\n0 0 100 0 0 2\n"
            "3 1 0 0 0 1 0 0 7 a.png\n0 0 2\n")
        (sparse / "points3D.txt").write_text(
            "100 1 0 2 10 20 30 0.25 9 0\n"
            "2 2 0 2 10 20 30 0.25 3 0 9 1\n")
        reconstruction = pycolmap.Reconstruction(str(sparse))
        for binary in (False, True):
            with self.subTest(binary=binary):
                if binary:
                    reconstruction.write_binary(str(sparse))
                model = pycolmap.Reconstruction(str(sparse))
                parsed = load_parser(model, real_paths=True)(tmp)
                self.assertEqual(parsed.image_names, ["a.png", "b.png"])
                np.testing.assert_array_equal(parsed.camtoworlds[:, 0, 3], [-1, -2])
                np.testing.assert_array_equal(parsed.points[:, 0], [2, 1])
                np.testing.assert_array_equal(parsed.point_indices["a.png"], [0])
                np.testing.assert_array_equal(parsed.point_indices["b.png"], [0, 1])
                np.testing.assert_array_equal(parsed.Ks_dict[7],
                                              [[4, 0, 4], [0, 5, 3], [0, 0, 1]])

    def test_unknown_camera_after_supported_camera_raises(self):
        image = lambda camera_id: types.SimpleNamespace(
            cam_from_world=lambda: types.SimpleNamespace(matrix=lambda: np.eye(4)[:3]),
            camera_id=camera_id,
        )
        camera = lambda model: types.SimpleNamespace(
            model=types.SimpleNamespace(name=model), calibration_matrix=lambda: np.eye(3),
            width=3, height=3,
        )
        manager = types.SimpleNamespace(
            images={0: image(0), 1: image(1)}, cameras={0: camera("PINHOLE"), 1: camera("UNKNOWN")},
        )
        parser = load_parser(manager)
        with self.subTest(), self.assertRaisesRegex(AssertionError, "Only perspective and fisheye"):
            parser("scene")

    def test_reconstruction_preserves_poses_and_sparse_track_indices(self):
        for method_pose in (False, True):
            with self.subTest(method_pose=method_pose), tempfile.TemporaryDirectory() as tmp:
                scene = Path(tmp)
                (scene / "sparse/0").mkdir(parents=True)
                (scene / "images").mkdir()

                def image(name, tx, method_pose=method_pose):
                    matrix = np.eye(4)[:3].copy()
                    matrix[0, 3] = tx
                    pose = types.SimpleNamespace(matrix=lambda: matrix)
                    return types.SimpleNamespace(name=name, camera_id=7,
                        cam_from_world=(lambda: pose) if method_pose else pose)

                def point(x, image_ids):
                    return types.SimpleNamespace(xyz=[x, 0, 2], error=0.25,
                        color=[10, 20, 30], track=types.SimpleNamespace(elements=[
                            types.SimpleNamespace(image_id=i) for i in image_ids]))

                class ReconstructionStub(types.SimpleNamespace):
                    points3D_reads = 0

                    @property
                    def points3D(self):
                        self.points3D_reads += 1
                        return self.point_data

                manager = ReconstructionStub(
                    images={9: image("b.png", 2), 3: image("a.png", 1)},
                    cameras={7: types.SimpleNamespace(model=types.SimpleNamespace(name="PINHOLE"),
                        calibration_matrix=lambda: np.array([[4., 0, 4], [0, 5, 3], [0, 0, 1]]),
                        width=8, height=6)},
                    point_data={100: point(1, [9]), 2: point(2, [3, 9])})
                parsed = load_parser(manager, real_paths=True)(tmp)
                self.assertEqual(manager.points3D_reads, 1)
                self.assertEqual(parsed.image_names, ["a.png", "b.png"])
                np.testing.assert_array_equal(parsed.camtoworlds[:, 0, 3], [-1, -2])
                np.testing.assert_array_equal(parsed.points[:, 0], [2, 1])
                np.testing.assert_array_equal(parsed.point_indices["a.png"], [0])
                np.testing.assert_array_equal(parsed.point_indices["b.png"], [0, 1])
                np.testing.assert_array_equal(parsed.Ks_dict[7], [[4, 0, 4], [0, 5, 3], [0, 0, 1]])
                self.assertEqual(parsed.points.dtype, np.float32)
                self.assertEqual(parsed.points_rgb.dtype, np.uint8)

    def test_crop_keeps_mask_and_reaches_lower_right(self):
        image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
        mask = np.arange(9).reshape(3, 3) % 2 == 0
        dataset = load_dataset(image)(parser_stub(mask), split="test", patch_size=2)
        with self.subTest(), patch.object(np.random, "randint", return_value=1) as randint:
            data = dataset[0]
            self.assertEqual(randint.call_count, 2)
            self.assertTrue(all(call.args == (0, 2) for call in randint.call_args_list))
            np.testing.assert_array_equal(data["image"].array, image[1:, 1:])
            np.testing.assert_array_equal(data["mask"].array, mask[1:, 1:])
            np.testing.assert_array_equal(data["K"].array[:2, 2], [0., 0.])

    def test_depths_allow_image_without_sparse_observations(self):
        image = np.zeros((3, 3, 3), dtype=np.uint8)
        dataset = load_dataset(image)(parser_stub(None), split="test", load_depths=True)
        with self.subTest():
            data = dataset[0]
            self.assertEqual(data["points"].array.shape, (0, 2))
            self.assertEqual(data["depths"].array.shape, (0,))


def _complete_colmap_binary_scene(tmp_path, model, size, intrinsics, distortion):
    from examples.datasets.colmap import Parser

    width, height = size
    fx, fy, cx, cy = intrinsics
    root = tmp_path / "colmap_scene"
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True)
    images = root / "images"
    images.mkdir()
    if model in (0, 2):
        params = [fx, cx, cy] + list(distortion)
    else:
        params = [fx, fy, cx, cy] + list(distortion)
    camera = struct.pack("<QIiQQ", 1, 1, model, width, height)
    (sparse / "cameras.bin").write_bytes(camera + struct.pack("<" + "d" * len(params), *params))
    record = struct.pack("<QI4d3dI", 1, 1, 1, 0, 0, 0, 0, 0, 0, 1)
    record += b"image.png" + bytes(1) + struct.pack("<QddQ", 1, cx, cy, 1)
    (sparse / "images.bin").write_bytes(record)
    points = struct.pack("<QQ3d3BdQII", 1, 1, 0, 0, 2, 128, 64, 32, 0, 1, 1, 0)
    (sparse / "points3D.bin").write_bytes(points)
    image = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    Image.fromarray(image).save(images / "image.png")
    parser = Parser(str(root), normalize=False, test_every=1)
    return parser, image


def _assert_complete_parser_sample(parser, image, maps, intrinsics, roi, mask):
    from examples.datasets.colmap import Dataset

    x, y, width, height = roi
    expected_image = image
    if maps is not None:
        mapx, mapy = maps
        np.testing.assert_allclose(parser.mapx_dict[1], mapx, rtol=0, atol=1e-6)
        np.testing.assert_allclose(parser.mapy_dict[1], mapy, rtol=0, atol=1e-6)
        assert list(parser.roi_undist_dict[1]) == list(roi)
        expected_image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        expected_image = expected_image[y:y + height, x:x + width]
    np.testing.assert_allclose(parser.Ks_dict[1], intrinsics, rtol=0, atol=1e-6)
    assert parser.imsize_dict[1] == (width, height)
    sample = Dataset(parser, split="test", load_depths=True)[0]
    np.testing.assert_array_equal(sample["image"].numpy(), expected_image.astype(np.float32))
    np.testing.assert_allclose(sample["K"].numpy(), intrinsics, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(sample["depths"].numpy(), [2.0])
    if mask is not None:
        np.testing.assert_array_equal(parser.mask_dict[1], mask)
        np.testing.assert_array_equal(sample["mask"].numpy(), mask)
    else:
        assert "mask" not in sample


@pytest.mark.parametrize("size", [(13, 9), (9, 7)])
@pytest.mark.parametrize("off_center", [False, True])
@pytest.mark.parametrize("distortion", [(0.0, 0.0, 0.0, 0.0), (0.08, -0.01, 0.001, -0.0001)])
def test_complete_parser_fisheye_projection(tmp_path, size, off_center, distortion):
    width, height = size
    cx, cy = (width * 0.3, height * 0.35) if off_center else ((width - 1) / 2, (height - 1) / 2)
    fx, fy = 5.0, 6.0
    parser, image = _complete_colmap_binary_scene(
        tmp_path, 5, size, (fx, fy, cx, cy), distortion
    )
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    rays = np.stack(((xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)), axis=-1)
    coefficients = np.array(distortion, dtype=np.float32).astype(np.float64)
    projected, _ = cv2.fisheye.projectPoints(
        rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), intrinsics, coefficients
    )
    projected = projected.reshape(height, width, 2).astype(np.float32)
    radius = np.linalg.norm(rays[..., :2], axis=-1)
    theta = np.arctan(radius)
    polynomial = sum(coefficient * theta ** (2 * index + 2) for index, coefficient in enumerate(coefficients))
    scale = np.divide(theta * (1 + polynomial), radius, out=np.ones_like(radius), where=radius > 0)
    analytic = np.stack((fx * rays[..., 0] * scale + cx, fy * rays[..., 1] * scale + cy), axis=-1)
    np.testing.assert_allclose(projected, analytic, rtol=0, atol=1e-6)
    mapx, mapy = projected[..., 0], projected[..., 1]
    valid = (mapx > 0) & (mapy > 0) & (mapx < width - 1) & (mapy < height - 1)
    rows, columns = np.nonzero(valid)
    x, y = columns.min(), rows.min()
    w, h = columns.max() + 1 - x, rows.max() + 1 - y
    cropped_intrinsics = intrinsics.copy()
    cropped_intrinsics[0, 2] -= x
    cropped_intrinsics[1, 2] -= y
    _assert_complete_parser_sample(
        parser, image, (mapx, mapy), cropped_intrinsics, (x, y, w, h), valid[y:y + h, x:x + w]
    )


@pytest.mark.parametrize("model", [0, 1, 2, 4])
def test_complete_parser_perspective_control(tmp_path, model):
    width, height = 13, 9
    fx, fy, cx, cy = 5.0, (5.0 if model in (0, 2) else 6.0), 4.0, 3.0
    distortion = {0: (), 1: (), 2: (0.01,), 4: (0.01, -0.001, 0.0, 0.0)}[model]
    parser, image = _complete_colmap_binary_scene(
        tmp_path, model, (width, height), (fx, fy, cx, cy), distortion
    )
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    maps, roi = None, (0, 0, width, height)
    if distortion:
        params = np.array([distortion[0], 0, 0, 0] if model == 2 else distortion, dtype=np.float32)
        new_intrinsics, roi = cv2.getOptimalNewCameraMatrix(
            intrinsics, params, (width, height), 0
        )
        maps = cv2.initUndistortRectifyMap(
            intrinsics, params, None, new_intrinsics, (width, height), cv2.CV_32FC1
        )
        intrinsics = new_intrinsics
    _assert_complete_parser_sample(parser, image, maps, intrinsics, roi, None)

if __name__ == "__main__":
    unittest.main()
