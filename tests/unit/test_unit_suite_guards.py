"""The unit suite's own guarantees, asserted rather than assumed.

The offline guard in conftest is invisible when it works and invisible when it
silently stops working — a test that forgot to inject an embedder would just
start downloading models again, and nothing would say so until the day
huggingface.co was unreachable (#221).
"""

import os


def test_model_downloads_are_blocked_for_unit_tests():
    """Every unit test runs with Hugging Face in offline mode.

    If this fails, the guard has been removed or its scope changed, and the
    suite is one forgotten injection away from depending on the network again.
    """
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
