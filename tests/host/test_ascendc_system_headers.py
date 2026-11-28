# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check default compiler options and opt-in header compatibility without CANN."""

import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CMAKE_EXECUTABLE = shutil.which("cmake")
if CMAKE_EXECUTABLE is not None:
    CMAKE_EXECUTABLE = str(Path(CMAKE_EXECUTABLE).resolve())


@unittest.skipUnless(CMAKE_EXECUTABLE, "CMake is required")
class AscendCSystemHeadersTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="gauss headers ")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.headers = [self.base / "include/c++/12",
                        self.base / "include/aarch64-linux-gnu/c++/12",
                        self.base / "lib/gcc/aarch64-linux-gnu/12/include"]
        for path in self.headers:
            path.mkdir(parents=True)
        for header in ("cstdint", "cfloat", "type_traits"):
            (self.headers[0] / header).touch()
        self.env = dict(os.environ)
        self.env.pop("CPLUS_INCLUDE_PATH", None)
        self.env.pop("ASCENDC_ENABLE_HOST_HEADERS", None)

    def configure(self, preamble, debug=True):
        script = self.base / "headers.cmake"
        debug_options = 'list(APPEND flags -g -O0)\n' if debug else ''
        script.write_text(
            f'cmake_minimum_required(VERSION 3.19)\n{preamble}\n'
            f'include("{ROOT.as_posix()}/cmake/ascendc_system_headers.cmake")\n'
            'ascendc_system_header_options(flags)\n'
            f'{debug_options}'
            f'set(ASCEND_AUTOGEN_PATH "{self.base.as_posix()}")\n'
            'set(CUSTOM_COMPILE_OPTIONS custom_compile_options.ini)\n'
            'file(WRITE "${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS}" "")\n'
            f'include("{ROOT.as_posix()}/cmake/func.cmake")\n'
            'if(flags)\n'
            '  add_ops_compile_options(ALL OPTIONS ${flags})\n'
            'endif()\n', encoding="utf-8")
        return subprocess.run([CMAKE_EXECUTABLE, "-P", str(script)], env=self.env,
                              capture_output=True, text=True)

    def test_gcc_discovery_persists_all_paths_and_debug_flags(self):
        # Simulate the host GCC query on either Windows or Linux.
        compiler = self.base / ("gcc.cmd" if os.name == "nt" else "gcc")
        if os.name == "nt":
            compiler.write_text(f'@echo off\necho {self.headers[2].as_posix()}\n')
        else:
            compiler.write_text(f'#!/bin/sh\nprintf "%s\\n" "{self.headers[2].as_posix()}"\n')
            compiler.chmod(0o700)
        result = self.configure(
            'set(ASCENDC_ENABLE_HOST_HEADERS ON CACHE BOOL "")\n'
            'set(CMAKE_CXX_COMPILER_ID GNU)\n'
            f'set(CMAKE_CXX_COMPILER "{compiler.as_posix()}")\n'
            f'set(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "{self.headers[0].as_posix()};'
            f'{self.headers[1].as_posix()};{self.base.as_posix()}")')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.check_generated_adapter()

    def check_generated_adapter(self, expected_options=None):
        """Run the real generator with no CPLUS_INCLUDE_PATH in its environment."""
        cfg = self.base / "ops.ini"
        cfg.write_text(
            '[Example]\nopFile.value=example\nopInterface.value=example\n'
            'input0.name=x\ninput0.paramType=required\ninput0.dtype=float\ninput0.format=ND\n'
            'output0.name=y\noutput0.paramType=required\noutput0.dtype=float\noutput0.format=ND\n',
            encoding="utf-8")
        (self.base / "example.cpp").touch()
        output = self.base / "dynamic"
        output.mkdir()
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "cmake/util/ascendc_impl_build.py"),
             str(cfg), "", "", str(self.base), str(output), str(self.base)],
            env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        tree = ast.parse((output / "example.py").read_text(encoding="utf-8"))
        assignment = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "custom_all_compile_options" in target_names:
                assignment = node
                break
        self.assertIsNotNone(assignment, "Generated adapter is missing custom_all_compile_options")
        options = ast.literal_eval(assignment.value)[0].get("__ALLSOC__", [])
        if expected_options is None:
            expected_options = [f"-I{p.as_posix()}" for p in self.headers] + ["-g", "-O0"]
        self.assertEqual(options, expected_options)

    def test_default_release_uses_toolchain_header_search(self):
        # Even a stale CPLUS_INCLUDE_PATH must not be baked into the adapters
        # unless the user explicitly enables the workaround.
        self.env["CPLUS_INCLUDE_PATH"] = "/missing/legacy/c++/12"
        result = self.configure(
            'set(CMAKE_CXX_COMPILER_ID GNU)\n'
            'function(execute_process)\n'
            '  message(FATAL_ERROR "Default build must not query host headers")\n'
            'endfunction()', debug=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("workaround disabled", result.stdout)
        self.assertEqual((self.base / "custom_compile_options.ini").read_text(), "")
        self.env.pop("CPLUS_INCLUDE_PATH")
        self.check_generated_adapter(expected_options=[])

    def test_default_debug_keeps_debug_flags_without_host_headers(self):
        result = self.configure("")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.check_generated_adapter(expected_options=["-g", "-O0"])

    def test_environment_off_does_not_enable_discovery(self):
        self.env["ASCENDC_ENABLE_HOST_HEADERS"] = "OFF"
        result = self.configure("", debug=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("workaround disabled", result.stdout)

    def test_environment_switch_enables_discovery_with_cached_default(self):
        self.env["ASCENDC_ENABLE_HOST_HEADERS"] = "ON"
        # This switch works even after a previous configure cached OFF.
        paths = ";".join(p.as_posix() for p in self.headers[:2])
        result = self.configure(
            'set(ASCENDC_ENABLE_HOST_HEADERS OFF CACHE BOOL "")\n'
            f'set(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "{paths}")')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.check_generated_adapter(
            expected_options=[f"-I{p.as_posix()}" for p in self.headers[:2]] + ["-g", "-O0"])

    def test_opt_in_without_headers_fails_during_configure(self):
        result = self.configure('set(ASCENDC_ENABLE_HOST_HEADERS ON CACHE BOOL "")')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot locate cstdint", result.stderr)

    def test_explicit_override(self):
        paths = ";".join(p.as_posix() for p in self.headers)
        # Fail immediately if explicit paths trigger any compiler query.
        result = self.configure(
            f'set(ASCENDC_SYSTEM_INCLUDE_DIRS "{paths}" CACHE STRING "")\n'
            'set(CMAKE_CXX_COMPILER_ID GNU)\n'
            f'set(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "{self.base.as_posix()}/unexpected/c++/99")\n'
            'function(execute_process)\n'
            '  message(FATAL_ERROR "Explicit paths must not query the compiler")\n'
            'endfunction()')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.check_generated_adapter()

    def test_reconfigure_resets_options_at_the_append_path(self):
        paths = ";".join(p.as_posix() for p in self.headers)
        options = self.base / "custom_compile_options.ini"
        options.write_text("ALL,,-Istale-arch35-path\n", encoding="utf-8")
        result = self.configure(f'set(ASCENDC_SYSTEM_INCLUDE_DIRS "{paths}" CACHE STRING "")')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("stale-arch35-path", options.read_text(encoding="utf-8"))
        self.assertEqual(len(options.read_text(encoding="utf-8").splitlines()), 1)
        self.check_generated_adapter()

    @unittest.skipIf(os.name == "nt", "CPLUS_INCLUDE_PATH uses Linux colon-separated paths")
    def test_environment_paths_are_persisted(self):
        self.env["ASCENDC_ENABLE_HOST_HEADERS"] = "ON"
        self.env["CPLUS_INCLUDE_PATH"] = ":".join(str(p) for p in self.headers)
        result = self.configure("")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.env.pop("CPLUS_INCLUDE_PATH")
        self.check_generated_adapter()

    def test_missing_header_fails_during_configure(self):
        (self.headers[0] / "cfloat").unlink()
        result = self.configure(
            f'set(ASCENDC_SYSTEM_INCLUDE_DIRS "{self.headers[0].as_posix()}" CACHE STRING "")')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot locate cfloat", result.stderr)
        self.assertIn("development headers", result.stderr)


if __name__ == "__main__":
    unittest.main()
