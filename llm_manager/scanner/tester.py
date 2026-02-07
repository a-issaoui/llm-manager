"""
GPU context testing logic.
"""

import atexit
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime

from .constants import (
    CLEANUP_DELAY_FAILURE,
    CLEANUP_DELAY_SUCCESS,
    CONTEXT_SAFETY_MARGIN,
    CONTEXT_TEST_WORKER,
    MIN_BINARY_SEARCH_GAP,
    STABILITY_REDUCTION_FACTOR,
    STABILITY_RETRIES,
    TEST_TIMEOUT,
)
from .types import ContextTestResult

logger = logging.getLogger(__name__)

# Global tracking for atexit cleanup
_temp_files: list[str] = []


def cleanup() -> None:
    """Clean up temporary files on exit"""
    for f in _temp_files:
        try:
            if os.path.exists(f):
                os.unlink(f)
        except OSError:
            pass
    _temp_files.clear()


atexit.register(cleanup)


class ContextTester:
    """Handles GPU context limit testing using isolated subprocesses."""

    def __init__(self) -> None:
        self.temp_files: list[str] = []

    def _cleanup_temp_files(self) -> None:
        for f in self.temp_files:
            try:
                if os.path.exists(f):
                    os.unlink(f)
            except OSError:
                pass
        self.temp_files.clear()

    def run_contxt_test(
        self,
        filepath: str,
        min_ctx: int,
        max_ctx: int,
        gpu_layers: int,
        gpu_device: int | None,
        kv_quant: bool,
        flash_attn: bool,
    ) -> ContextTestResult:
        """Binary search for maximum stable context with progress logging"""

        def test_single(ctx: int, ctx_label: str = "") -> tuple[bool, str | None]:
            """Run isolated test at specific context"""
            logger.info(f"    Testing {ctx:,} tokens... {ctx_label}")

            worker_file = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False
                ) as f:
                    f.write(CONTEXT_TEST_WORKER)
                    worker_file = f.name
                    self.temp_files.append(worker_file)
                    _temp_files.append(
                        worker_file
                    )  # Keep global tracking for atexit safety

                args = {
                    "model_path": str(filepath),
                    "ctx_size": ctx,
                    "gpu_layers": gpu_layers,
                    "flash_attn": flash_attn,
                    "kv_quant": kv_quant,
                    "gpu_device": gpu_device,
                }

                result = subprocess.run(
                    [sys.executable, worker_file, json.dumps(args)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=TEST_TIMEOUT,
                )

                _delay = (
                    CLEANUP_DELAY_SUCCESS
                    if result.returncode == 0
                    else CLEANUP_DELAY_FAILURE
                )
                time.sleep(_delay)

                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    return data.get("success", False), data.get("error")
                return False, result.stderr[:200]

            except subprocess.TimeoutExpired:
                return False, "Timeout"
            except Exception as e:
                return False, str(e)[:200]
            finally:
                if worker_file and os.path.exists(worker_file):
                    try:
                        os.unlink(worker_file)
                        if worker_file in self.temp_files:
                            self.temp_files.remove(worker_file)
                        if worker_file in _temp_files:
                            try:
                                _temp_files.remove(worker_file)
                            except ValueError:
                                pass
                    except (OSError, ValueError):
                        pass

        # Test minimum first
        logger.info(f"  [Phase 1/3] Testing minimum context {min_ctx:,}...")
        success, err = test_single(min_ctx, "(minimum)")
        if not success:
            error_msg = err or "Unknown error"
            if "n_ctx_per_seq" in error_msg and "n_ctx_train" in error_msg:
                match = re.search(r"n_ctx_train \((\d+)\)", error_msg)
                required_ctx = int(match.group(1)) if match else "unknown"
                error_msg = (
                    f"Model requires min context {required_ctx} (HW insufficient)"
                )

            return ContextTestResult(
                max_context=0,
                recommended_context=0,
                buffer_tokens=0,
                buffer_percent=20,
                tested=True,
                verified_stable=False,
                error=error_msg,
                timestamp=datetime.now().isoformat(),
                confidence=0.0,
            )

        logger.info(f"    ✓ Minimum {min_ctx:,} works")

        # Binary search with progress logging
        logger.info("  [Phase 2/3] Binary searching optimal context...")
        low, high = min_ctx, max_ctx
        best_working = min_ctx
        iteration = 0

        while high - low > MIN_BINARY_SEARCH_GAP:
            iteration += 1
            gap = high - low

            if gap > 32768:
                step = 4096
            elif gap > 16384:
                step = 2048
            else:
                step = 1024

            mid = (low + high) // 2
            mid = (mid // step) * step

            if mid <= low:
                mid = low + step
            if mid >= high:
                mid = high - step

            if mid <= low or mid >= high or mid == best_working:
                logger.debug(f"    Binary search converged at gap={gap:,}")
                break

            success, err = test_single(
                mid, f"(iteration {iteration}, gap {gap:,})"
            )

            if success:
                best_working = mid
                low = mid
                logger.info(f"      ✓ {mid:,} works (new best)")
            else:
                high = mid
                logger.info(f"      ✗ {mid:,} failed")

        logger.info(f"  Best found: {best_working:,} tokens")

        # Stability verification with progress
        logger.info(f"  [Phase 3/3] Verifying stability at {best_working:,}...")
        verified = True
        attempts = 0
        max_attempts = STABILITY_RETRIES + 2

        while attempts < max_attempts and best_working > min_ctx:
            attempts += 1
            ctx_label = f"(stability check {attempts}/{STABILITY_RETRIES})"
            success, err = test_single(best_working, ctx_label)

            if success:
                success2, _ = test_single(best_working, "(confirmation)")
                if success2:
                    verified = True
                    break

            verified = False
            reduction = int(best_working * STABILITY_REDUCTION_FACTOR)
            new_working = max(min_ctx, best_working - reduction)

            if new_working >= best_working:
                break

            best_working = new_working
            logger.warning(f"    Unstable, reduced to {best_working:,}")

        recommended = int(best_working * CONTEXT_SAFETY_MARGIN)

        confidence = 1.0
        if not verified:
            confidence *= 0.7
        if best_working < max_ctx * 0.2:
            confidence *= 0.8

        logger.info(
            f"  Result: {best_working:,} max → {recommended:,} recommended "
            f"({best_working - recommended:,} buffer) "
            f"[{'Stable' if verified else 'Unstable'}, {confidence * 100:.0f}% confidence]"
        )

        return ContextTestResult(
            max_context=best_working,
            recommended_context=recommended,
            buffer_tokens=best_working - recommended,
            buffer_percent=int((1 - CONTEXT_SAFETY_MARGIN) * 100),
            tested=True,
            verified_stable=verified,
            test_config={
                "kv_quant": "q4_0" if kv_quant else "f16",
                "flash_attn": flash_attn,
                "gpu_layers": gpu_layers,
            },
            timestamp=datetime.now().isoformat(),
            confidence=confidence,
        )
