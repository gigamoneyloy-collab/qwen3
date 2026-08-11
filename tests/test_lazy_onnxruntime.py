# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class LazyOnnxRuntimeTest(unittest.TestCase):
    def _run_python(self, code):
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT), env.get("PYTHONPATH", "")]
        )
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            check=True,
            cwd=REPO_ROOT,
            env=env,
        )

    def test_12hz_import_does_not_load_onnxruntime(self):
        self._run_python(
            """
            import sys

            from qwen_tts import Qwen3TTSTokenizer
            from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
                Qwen3TTSTokenizerV2Model,
            )

            assert Qwen3TTSTokenizer.__name__ == "Qwen3TTSTokenizer"
            assert Qwen3TTSTokenizerV2Model.__name__ == "Qwen3TTSTokenizerV2Model"
            assert "onnxruntime" not in sys.modules
            """
        )

    def test_25hz_xvector_extractor_loads_onnxruntime_on_demand(self):
        self._run_python(
            """
            import sys
            import types

            from qwen_tts.core.tokenizer_25hz.vq import speech_vq

            assert "onnxruntime" not in sys.modules
            calls = []


            class SessionOptions:
                pass


            class GraphOptimizationLevel:
                ORT_ENABLE_ALL = object()


            class Transformer:
                def norm(self, **kwargs):
                    calls.append(("norm", kwargs))


            fake_ort = types.ModuleType("onnxruntime")
            fake_ort.SessionOptions = SessionOptions
            fake_ort.GraphOptimizationLevel = GraphOptimizationLevel


            def inference_session(path, *, sess_options, providers):
                calls.append(("session", path, sess_options, providers))
                return object()


            fake_ort.InferenceSession = inference_session
            sys.modules["onnxruntime"] = fake_ort
            speech_vq.sox.Transformer = Transformer

            extractor = speech_vq.XVectorExtractor("campplus.onnx")
            assert extractor.ort_session is not None
            assert calls[0][0] == "session"
            assert calls[0][1] == "campplus.onnx"
            assert calls[0][3] == ["CPUExecutionProvider"]
            assert calls[1] == ("norm", {"db_level": -6})
            """
        )


if __name__ == "__main__":
    unittest.main()
