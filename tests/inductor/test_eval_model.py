# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""``tools/cost_model/eval_model.py`` is run as a script, never imported as a package
module, so nothing else in CI notices when it stops loading."""

import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOOL_DIR = os.path.join(_ROOT, "tools", "cost_model")


def test_eval_model_is_importable(monkeypatch):
    # The script puts its own directory and the repo root first on sys.path and
    # imports its siblings (``records``) as top-level modules. Undo both, so no
    # later test in this worker resolves an import against the tool directory.
    monkeypatch.setattr(sys, "path", list(sys.path))
    before = set(sys.modules)
    try:
        path = os.path.join(_TOOL_DIR, "eval_model.py")
        spec = importlib.util.spec_from_file_location("eval_model", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert callable(mod.main)
    finally:
        for name in set(sys.modules) - before:
            origin = getattr(sys.modules[name], "__file__", None) or ""
            if os.path.dirname(os.path.abspath(origin)) == _TOOL_DIR:
                del sys.modules[name]
