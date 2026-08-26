"""The paged KV runtime's layering rule, enforced rather than documented.

Two packages, two allowances:

  * ``kv_common`` -- the neutral vocabulary -- may import the standard library
    and ``numpy``, and nothing else.
  * ``kv_control`` -- the semantic control plane -- may also import
    ``kv_common``.

Neither may import ``jax``, ``jax_aiter``, ``maxtext.layers`` or
``maxtext.models``. Vendor kernel ABIs are reached only from ``kv_execution``,
one layer up.

Both properties this buys are load-bearing, which is why the rule is checked and
not merely written down. The layers stay testable with no accelerator present,
so their logic is covered by ordinary CI rather than by a GPU job. And if a
second consumer ever wants the control plane, extracting it is a directory move
with no vendor dependency to untangle first.

Checked statically over the AST, so a violation is reported without importing
anything, and one subprocess check confirms the static rule matches what the
interpreter actually loads.

Copyright 2026 Advanced Micro Devices, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import ast
import dataclasses
import pathlib
import subprocess
import sys
import unittest

_INFERENCE_DIR = pathlib.Path(__file__).resolve().parents[2] / "src" / "maxtext" / "inference"

# A package importing any of these has broken the rule outright, and saying so
# by name gives a better failure than "unexpected import".
FORBIDDEN_ROOTS = frozenset({"jax", "jaxlib", "flax", "torch", "jax_aiter", "tensorflow"})

# Everything either layer legitimately needs. Extend only for something that is
# genuinely stdlib or numpy.
ALLOWED_ROOTS = frozenset(
    {
        "__future__",
        "collections",
        "dataclasses",
        "enum",
        "hashlib",
        "math",
        "numpy",
        "typing",
    }
)


@dataclasses.dataclass(frozen=True)
class _Layer:
  """One package and the dotted prefixes under `maxtext` it may reach."""

  package: str
  allowed_maxtext: frozenset[str]

  @property
  def directory(self) -> pathlib.Path:
    return _INFERENCE_DIR / self.package.rsplit(".", 1)[1]

  def permits_maxtext(self, module: str) -> bool:
    return any(module == prefix or module.startswith(prefix + ".") for prefix in self.allowed_maxtext)


_KV_COMMON = "maxtext.inference.kv_common"
_KV_CONTROL = "maxtext.inference.kv_control"
# Deliberately not a checked layer: `kv_execution` is where jax, MaxText config
# and the vendor backend legitimately live. What matters is that nothing below it
# imports it, which the test below asserts.
_KV_EXECUTION = "maxtext.inference.kv_execution"

LAYERS = (
    _Layer(package=_KV_COMMON, allowed_maxtext=frozenset({_KV_COMMON})),
    _Layer(package=_KV_CONTROL, allowed_maxtext=frozenset({_KV_CONTROL, _KV_COMMON})),
)


def _imports(path: pathlib.Path, layer: _Layer) -> set[str]:
  """Root packages imported by `path`, with permitted `maxtext` imports elided.

  A `maxtext` import the layer does not permit is reported under its full dotted
  name rather than as the root `maxtext`, so the failure message names the actual
  violation instead of making the reader go and find it.
  """
  roots: set[str] = set()
  for node in ast.walk(ast.parse(path.read_text())):
    if isinstance(node, ast.Import):
      for alias in node.names:
        if layer.permits_maxtext(alias.name):
          continue
        roots.add(alias.name if alias.name.startswith("maxtext") else alias.name.split(".")[0])
    elif isinstance(node, ast.ImportFrom):
      if node.level or not node.module:  # a relative import stays inside the package
        continue
      if layer.permits_maxtext(node.module):
        continue
      roots.add(node.module if node.module.startswith("maxtext") else node.module.split(".")[0])
  return roots


class ImportRuleTest(unittest.TestCase):
  """Each layer imports only what its position in the stack allows."""

  def test_layer_directories_exist(self):
    """Guards against the rule passing vacuously if a package is moved."""
    for layer in LAYERS:
      self.assertTrue(layer.directory.is_dir(), f"{layer.package} not found at {layer.directory}")
      self.assertTrue(sorted(layer.directory.glob("*.py")), f"{layer.package} has no modules to check")

  def test_no_forbidden_imports(self):
    for layer in LAYERS:
      for path in sorted(layer.directory.glob("*.py")):
        forbidden = _imports(path, layer) & FORBIDDEN_ROOTS
        self.assertEqual(
            forbidden,
            set(),
            f"{layer.package}/{path.name} imports {sorted(forbidden)}, which the import rule forbids",
        )

  def test_no_unexpected_imports(self):
    for layer in LAYERS:
      for path in sorted(layer.directory.glob("*.py")):
        unexpected = _imports(path, layer) - ALLOWED_ROOTS
        self.assertEqual(
            unexpected,
            set(),
            f"{layer.package}/{path.name} imports {sorted(unexpected)}; extend ALLOWED_ROOTS only if "
            f"the addition is genuinely stdlib or numpy, and never to admit a maxtext module",
        )

  def test_kv_common_does_not_reach_up_into_kv_control(self):
    """The dependency runs one way. A cycle here would defeat extraction."""
    common = LAYERS[0]
    for path in sorted(common.directory.glob("*.py")):
      self.assertNotIn(
          _KV_CONTROL,
          _imports(path, common),
          f"kv_common/{path.name} imports kv_control, inverting the layering",
      )

  def test_neither_lower_layer_reaches_up_into_kv_execution(self):
    """`kv_execution` is where jax and the vendor backend live.

    It exists precisely to hold the couplings the two layers below refuse, so a
    single import in this direction would put jax back into the control plane's
    dependency graph and undo the whole split. Named separately from the
    allow-list check because this is the failure most likely to be introduced by
    someone reaching for something convenient.
    """
    for layer in LAYERS:
      for path in sorted(layer.directory.glob("*.py")):
        self.assertNotIn(
            _KV_EXECUTION,
            _imports(path, layer),
            f"{layer.package}/{path.name} imports kv_execution, inverting the layering",
        )

  def test_package_inits_only_re_export_their_own_modules(self):
    for layer in LAYERS:
      tree = ast.parse((layer.directory / "__init__.py").read_text())
      for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
          self.assertTrue(
              node.module.startswith(layer.package),
              f"{layer.package}/__init__.py imports from {node.module}, outside its own package",
          )

  def test_importing_the_control_plane_does_not_load_a_framework(self):
    """What the static rule is actually for, confirmed against the interpreter.

    A fresh interpreter, because by the time this test runs pytest's own
    collection has already imported jax, so `sys.modules` in-process proves
    nothing.
    """
    code = (
        "import sys, maxtext.inference.kv_control, maxtext.inference.kv_common;"
        "print(sorted({m.split('.')[0] for m in sys.modules} & "
        "{'jax', 'jaxlib', 'flax', 'torch', 'tensorflow'}))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    self.assertEqual(result.returncode, 0, f"importing the KV layers failed:\n{result.stderr}")
    self.assertEqual(result.stdout.strip(), "[]", f"a framework was loaded: {result.stdout.strip()}")


if __name__ == "__main__":
  unittest.main()
