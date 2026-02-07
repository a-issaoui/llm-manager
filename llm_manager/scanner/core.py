"""
Core scanner logic and high-level API.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .constants import (
    CONTEXT_SAFETY_MARGIN,
    DEFAULT_UNTESTED_CONTEXT,
    __version__,
)
from .detector import CapabilityDetector
from .metadata import MetadataExtractor
from .reader import GGUFReader, get_file_hash
from .tester import ContextTester
from .types import (
    ContextTestResult,
    ModelCapabilities,
    ModelEntry,
    ModelSpecs,
)

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Import llm_manager components
try:
    from ..exceptions import ValidationError
    from ..models import ModelRegistry

    _HAS_LLM_MANAGER = True
except ImportError:
    _HAS_LLM_MANAGER = False
    # Mock for standalone usage
    class ValidationError(Exception):  # type: ignore[no-redef]
        pass

    class ModelRegistry:  # type: ignore[no-redef]
        def __init__(self, path: str) -> None:
            pass

logger = logging.getLogger(__name__)


class PerfectScanner:
    """Low-level GGUF scanner with metadata extraction and context testing."""

    def __init__(self) -> None:
        self.results: dict[str, ModelEntry] = {}
        self.mmproj_files: dict[str, dict[str, Any]] = {}
        self.stats = {
            "total": 0,
            "parsed": 0,
            "failed": 0,
            "context_tested": 0,
            "context_skipped": 0,
            "context_failed": 0,
        }
        self._save_counter = 0
        self.context_tester = ContextTester()

    def scan_mmproj(self, filepath: str) -> dict[str, Any] | None:
        """Extract vision adapter metadata"""
        metadata = GGUFReader.extract_metadata(filepath)
        if not metadata:
            return None

        arch = metadata.get("general.architecture", "clip")
        if arch != "clip" and "clip" not in arch:
            return None

        mmproj = {
            "architecture": arch,
            "quantization": MetadataExtractor.detect_quantization(
                metadata, Path(filepath).name
            ),
            "file_size_mb": Path(filepath).stat().st_size // (1024 * 1024),
        }

        vision_params = {
            "vision_embedding_length": "clip.vision_embedding_length",
            "projection_dim": "clip.projection_dim",
            "patch_size": "clip.patch_size",
            "image_size": "clip.image_size",
        }
        for param, key in vision_params.items():
            if key in metadata:
                val = metadata[key]
                if not (isinstance(val, str) and val.startswith("<array:")):
                    mmproj[param] = val

        return mmproj

    def find_parent_model(self, mmproj_name: str) -> str | None:
        """Fuzzy match mmproj to base model"""
        base = (
            mmproj_name.replace("mmproj-", "")
            .replace("-mmproj", "")
            .replace(".gguf", "")
        )
        base = re.sub(
            r"-(f16|f32|q\d+[_k]*)", "", base, flags=re.IGNORECASE
        ).lower()
        base_tokens = [t for t in re.split(r"[-_.]", base) if t]

        best_match = None
        best_score = 0.0

        for filename, entry in self.results.items():
            if "mmproj" in filename or entry.capabilities.embed:
                continue

            model_lower = filename.lower()
            model_tokens = [t for t in re.split(r"[-_.]", model_lower) if t]

            score = sum(
                1.5 if len(t) > 3 else 1.0 for t in base_tokens if t in model_tokens
            )

            if entry.capabilities.vision:
                score += 2.0

            if score > best_score and score >= 3.0:
                best_score = score
                best_match = filename

        return best_match

    def save_atomic(self, filepath: str, data: dict[str, Any]) -> None:
        """Atomic write with backup"""
        if os.path.exists(filepath):
            try:
                shutil.copy2(filepath, f"{filepath}.backup")
            except (OSError, shutil.Error):
                pass

        dir_name = os.path.dirname(filepath) or "."
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, filepath)
        except Exception as e:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            raise e

    def scan_and_test(
        self,
        folder: str,
        output: str,
        test_context: bool = False,
        resume: bool = False,
        min_ctx: int = 8192,
        max_ctx: int = 131072,
        gpu_layers: int = -1,
        gpu_device: int | None = None,
        kv_quant: bool = True,
        flash_attn: bool = True,
        model_filter: str | None = None,
    ) -> None:
        """Main scanning and testing workflow"""

        path = Path(folder)
        if not path.exists():
            logger.error(f"Directory not found: {folder}")
            return

        gguf_files: list[Path] = list(path.rglob("*.gguf"))
        if not gguf_files:
            logger.error("No GGUF files found")
            return

        if model_filter:
            gguf_files = [
                f for f in gguf_files if model_filter.lower() in f.name.lower()
            ]

        self.stats["total"] = len(gguf_files)

        if resume and os.path.exists(output):
            try:
                with open(output, encoding="utf-8") as f:
                    existing = json.load(f)

                for fname, data in existing.items():
                    ctx_data = data.get("specs", {}).get("context_test", {})
                    if isinstance(ctx_data, dict):
                        context_test = ContextTestResult(**ctx_data)
                    else:
                        context_test = ContextTestResult(
                            max_context=0,
                            recommended_context=0,
                            buffer_tokens=0,
                            buffer_percent=20,
                            tested=False,
                            verified_stable=False,
                        )

                    specs_data = {
                        k: v
                        for k, v in data.get("specs", {}).items()
                        if k not in ("context_test", "optimized_kv_quant")
                    }

                    # Handle missing fields in legacy specs if any
                    specs = ModelSpecs(**specs_data)
                    specs.context_test = context_test
                    specs.optimized_kv_quant = data.get("specs", {}).get(
                        "optimized_kv_quant", False
                    )

                    self.results[fname] = ModelEntry(
                        specs=specs,
                        capabilities=ModelCapabilities(**data.get("capabilities", {})),
                        prompt=data.get("prompt", {}),
                        mmproj=data.get("mmproj"),
                        path=data.get("path", ""),
                    )

                logger.info(f"Loaded {len(self.results)} existing results")
            except Exception as e:
                logger.warning(f"Could not resume: {e}")

        logger.info(f"\n{'=' * 70}")
        logger.info(f"Perfect GGUF Scanner v{__version__}")
        logger.info(f"{'=' * 70}")
        logger.info(f"Found {len(gguf_files)} GGUF files")
        logger.info(f"Context testing: {'Enabled' if test_context else 'Disabled'}")
        if test_context:
            logger.info(f"KV Quantization: {'Q4_0' if kv_quant else 'FP16'}")
        logger.info(f"{'=' * 70}\n")

        regular_files: list[Path] = []
        mmproj_files: list[Path] = []

        for gguf_file in sorted(gguf_files):
            target = (
                mmproj_files if "mmproj" in gguf_file.name.lower() else regular_files
            )
            target.append(gguf_file)

        for idx, filepath in enumerate(regular_files, 1):
            fname = filepath.name

            if fname in self.results:
                logger.info(f"[{idx}/{len(regular_files)}] {fname} - Already scanned")
            else:
                logger.info(f"[{idx}/{len(regular_files)}] Scanning {fname}...")

                metadata = GGUFReader.extract_metadata(str(filepath))
                if not metadata:
                    self.stats["failed"] += 1
                    continue

                arch = metadata.get("general.architecture", "unknown")
                specs = MetadataExtractor.extract_specs(str(filepath), metadata, arch)
                capabilities = CapabilityDetector.detect(fname, arch, metadata)
                template = metadata.get("tokenizer.chat_template", "")
                if isinstance(template, (list, dict)):
                    if isinstance(template, list):
                        template = str(template[0])
                    else:
                        template = str(template.get("default", ""))

                entry = ModelEntry(
                    specs=specs,
                    capabilities=capabilities,
                    prompt={"template": template},
                    path=str(filepath),
                )

                self.results[fname] = entry
                self.stats["parsed"] += 1

                caps = [k for k, v in vars(capabilities).items() if v]
                logger.info(
                    f"  ├─ {specs.architecture} | {specs.parameters_b}B | "
                    f"{specs.quantization} | {specs.context_window // 1024}k"
                )
                logger.info(
                    f"  └─ Capabilities: {', '.join(caps) if caps else 'none'}"
                )

            if test_context and not self.results[fname].capabilities.embed:
                existing_test = self.results[fname].specs.context_test

                if existing_test.tested and resume:
                    self.stats["context_skipped"] += 1
                    _ctx = existing_test.recommended_context
                    logger.info(f"  └─ Context: Already tested ({_ctx})")
                    continue

                current_hash = get_file_hash(str(filepath))
                if (
                    existing_test.tested
                    and self.results[fname].specs.file_hash == current_hash
                ):
                    self.stats["context_skipped"] += 1
                    logger.info("  └─ Context: Skipped (hash match)")
                    continue

                try:
                    result = self.context_tester.run_contxt_test(
                        str(filepath),
                        min_ctx=min_ctx,
                        max_ctx=min(max_ctx, self.results[fname].specs.context_window),
                        gpu_layers=gpu_layers,
                        gpu_device=gpu_device,
                        kv_quant=kv_quant,
                        flash_attn=flash_attn,
                    )

                    self.results[fname].specs.context_test = result
                    self.results[fname].specs.optimized_kv_quant = kv_quant
                    self.results[fname].specs.file_hash = current_hash

                    if result.error:
                        self.stats["context_failed"] += 1
                        logger.warning(
                            f"  └─ Context test failed: {result.error[:80]}"
                        )
                    else:
                        self.stats["context_tested"] += 1
                        if result.confidence < 1.0:
                            confidence_str = f" [{result.confidence * 100:.0f}%]"
                        else:
                            confidence_str = ""
                        logger.info(
                            f"  └─ Context: {result.max_context} max → "
                            f"{result.recommended_context} stable{confidence_str}"
                        )

                    self._save_counter += 1
                    if self._save_counter % 5 == 0:
                        self.save_results(output)

                except KeyboardInterrupt:
                    logger.info("\nInterrupted - saving progress...")
                    self.save_results(output)
                    raise
                except Exception as e:
                    self.stats["context_failed"] += 1
                    logger.error(f"  └─ Context test error: {e}")

        if mmproj_files:
            logger.info(f"\nLinking {len(mmproj_files)} vision adapters...")
            for mmproj_path in mmproj_files:
                mmproj_data = self.scan_mmproj(str(mmproj_path))
                if not mmproj_data:
                    continue

                parent = self.find_parent_model(mmproj_path.name)
                if parent and parent in self.results:
                    self.results[parent].mmproj = mmproj_data

                    parent_lower = parent.lower()
                    if "whisper" in parent_lower:
                        self.results[parent].capabilities.audio_in = True
                    else:
                        self.results[parent].capabilities.vision = True

                    logger.info(f"  {mmproj_path.name} → {parent}")

        self.save_results(output)
        self.print_summary()

    def save_results(self, output: str) -> None:
        """Serialize results to JSON with proper defaults for untested models"""
        data = {}
        for fname, entry in self.results.items():
            specs_dict = asdict(entry.specs)

            if not specs_dict["context_test"]["tested"]:
                native = specs_dict["context_window"]
                default_max = min(DEFAULT_UNTESTED_CONTEXT, native // 2)
                specs_dict["context_test"]["max_context"] = default_max
                _rec = int(default_max * CONTEXT_SAFETY_MARGIN)
                specs_dict["context_test"]["recommended_context"] = _rec
                _rec_ctx = specs_dict["context_test"]["recommended_context"]
                specs_dict["context_test"]["buffer_tokens"] = default_max - _rec_ctx

            data[fname] = {
                "specs": specs_dict,
                "capabilities": vars(entry.capabilities),
                "prompt": entry.prompt,
                "mmproj": entry.mmproj,
                "path": entry.path,
            }

        self.save_atomic(output, data)

    def print_summary(self) -> None:
        """Print final statistics"""
        logger.info(f"\n{'=' * 70}")
        logger.info("SCAN COMPLETE")
        logger.info(f"{'=' * 70}")
        logger.info(f"Total files:         {self.stats['total']}")
        logger.info(f"Parsed successfully: {self.stats['parsed']}")
        logger.info(f"Failed to parse:     {self.stats['failed']}")
        if self.stats["context_tested"] or self.stats["context_failed"]:
            logger.info(f"Context tested:      {self.stats['context_tested']}")
            logger.info(f"Context skipped:     {self.stats['context_skipped']}")
            logger.info(f"Context failed:      {self.stats['context_failed']}")
        logger.info(f"{'=' * 70}")


class ModelScanner:
    """
    High-level scanner API for llm_manager integration.
    Wraps PerfectScanner with simplified interface.
    """

    def __init__(
        self,
        models_dir: str | Path,
        registry_file: str = "models.json",
        config_file: str = "llm_manager.yaml",
    ):
        self.models_dir = Path(models_dir)
        self.registry_file = self.models_dir / registry_file
        self.config_file = self.models_dir / config_file
        self._scanner = PerfectScanner()

    def scan_and_save(
        self,
        test_context: bool = False,
        resume: bool = False,
        min_context: int = 8192,
        max_context: int = 131072,
        gpu_layers: int = -1,
        gpu_device: int | None = None,
        kv_quant: bool = True,
        flash_attn: bool = True,
        model_filter: str | None = None,
    ) -> dict[str, Any]:
        if not self.models_dir.exists():
            if _HAS_LLM_MANAGER:
                raise ValidationError(f"Models directory not found: {self.models_dir}")
            else:
                raise FileNotFoundError(
                    f"Models directory not found: {self.models_dir}"
                )

        logger.info(f"Scanning {self.models_dir}...")

        self._scanner.scan_and_test(
            folder=str(self.models_dir),
            output=str(self.registry_file),
            test_context=test_context,
            resume=resume,
            min_ctx=min_context,
            max_ctx=max_context,
            gpu_layers=gpu_layers,
            gpu_device=gpu_device,
            kv_quant=kv_quant,
            flash_attn=flash_attn,
            model_filter=model_filter,
        )

        if YAML_AVAILABLE:
            self._generate_config()

        return {
            "registry_file": str(self.registry_file),
            "config_file": str(self.config_file) if YAML_AVAILABLE else None,
            "stats": self._scanner.stats,
            "models_found": len(self._scanner.results),
        }

    def _generate_config(self) -> None:
        if not self._scanner.results:
            logger.warning("No models scanned, skipping config generation")
            return

        config: dict[str, Any] = {
            "llm_manager": {
                "version": "5.0.0",
                "models_dir": str(self.models_dir),
                "registry_file": self.registry_file.name,
            },
            "scan_results": {
                "total_models": len(self._scanner.results),
                "context_tested": self._scanner.stats.get("context_tested", 0),
            },
            "models": {},
            "recommended_defaults": {},
        }

        total_params = []
        max_contexts = []

        for fname, entry in self._scanner.results.items():
            specs = entry.specs

            model_config = {
                "filename": fname,
                "architecture": specs.architecture,
                "parameters_b": specs.parameters_b,
                "quantization": specs.quantization,
                "context_window": specs.context_window,
                "capabilities": {
                    "vision": entry.capabilities.vision,
                    "embedding": entry.capabilities.embed,
                    "reasoning": entry.capabilities.reasoning,
                    # Note: multilingual mapped to reasoning (preserved from original)
                    "multilingual": entry.capabilities.reasoning,
                    "tools": entry.capabilities.tools,
                },
            }
            # Checking original scanner.py line 1591:
            # Yes, it was a typo (multilingual mapped to reasoning).
            # Preserved for exact behavior matching.
            # Not fixed - ModelCapabilities lacks multilingual.
            # Keeping to minimize regression risk.


            if specs.context_test and specs.context_test.tested:
                model_config[
                    "recommended_context"
                ] = specs.context_test.recommended_context
                model_config["max_tested_context"] = specs.context_test.max_context
                max_contexts.append(specs.context_test.recommended_context)

            if specs.parameters_b:
                total_params.append(specs.parameters_b)

            config["models"][fname] = model_config

        if max_contexts:
            config["recommended_defaults"]["typical_context"] = int(
                sum(max_contexts) / len(max_contexts)
            )
            config["recommended_defaults"]["max_safe_context"] = min(max_contexts)

        if total_params:
            avg_params = sum(total_params) / len(total_params)
            config["recommended_defaults"]["avg_parameters_b"] = round(avg_params, 2)

        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Generated config: {self.config_file}")
        except Exception as e:
            logger.warning(f"Failed to write llm_manager.yaml: {e}")

    def get_registry(self) -> Any:
        if not _HAS_LLM_MANAGER:
            raise RuntimeError(
                "llm_manager not available, cannot create ModelRegistry"
            )

        if not self.registry_file.exists():
            raise ValidationError(f"Registry not found: {self.registry_file}")

        return ModelRegistry(str(self.models_dir))

    def quick_scan(self) -> dict[str, Any]:
        return self.scan_and_save(test_context=False)


def scan_models(
    models_dir: str | Path, test_context: bool = False, **kwargs: Any
) -> dict[str, Any]:
    scanner = ModelScanner(models_dir)
    return scanner.scan_and_save(test_context=test_context, **kwargs)


async def scan_models_async(
    models_dir: str | Path, test_context: bool = False, **kwargs: Any
) -> dict[str, Any]:
    return await asyncio.to_thread(scan_models, models_dir, test_context, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Perfect GGUF Scanner v{__version__} - Metadata & Context Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Metadata scan only
  python -m llm_manager.scanner ./models

  # Full scan with context testing (Q4_0 KV cache)
  python -m llm_manager.scanner ./models --test-context

  # Resume interrupted scan
  python -m llm_manager.scanner ./models --test-context --resume
''',
    )

    parser.add_argument("folder", help="Directory containing GGUF files")
    parser.add_argument(
        "-o", "--output", default="models.json", help="Output JSON file"
    )
    parser.add_argument(
        "--test-context", action="store_true", help="Enable GPU context testing"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from previous results"
    )
    parser.add_argument(
        "--min-context", type=int, default=8192, help="Minimum context to test"
    )
    parser.add_argument(
        "--max-context", type=int, default=131072, help="Maximum context to test"
    )
    parser.add_argument(
        "--gpu-layers", type=int, default=-1, help="GPU layers (-1 = all)"
    )
    parser.add_argument("--gpu-device", type=int, help="GPU device index")
    parser.add_argument(
        "--no-kv-quant", action="store_true", help="Use FP16 for KV cache (more VRAM)"
    )
    parser.add_argument(
        "--no-flash-attn", action="store_true", help="Disable flash attention"
    )
    parser.add_argument("--model-filter", help="Filter models by name substring")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout
    )

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.test_context:
        try:
            import llama_cpp

            logger.info(
                f"Using llama-cpp-python {getattr(llama_cpp, '__version__', 'unknown')}"
            )
        except ImportError:
            logger.error("llama-cpp-python required for context testing")
            logger.error("Install: pip install llama-cpp-python")
            return 1

        try:
            import torch

            if torch.cuda.is_available():
                dev = args.gpu_device or 0
                logger.info(f"GPU {dev}: {torch.cuda.get_device_name(dev)}")
            else:
                logger.warning("CUDA not available - testing may fail")
        except ImportError:
            pass

    scanner = PerfectScanner()
    scanner.scan_and_test(
        folder=args.folder,
        output=args.output,
        test_context=args.test_context,
        resume=args.resume,
        min_ctx=args.min_context,
        max_ctx=args.max_context,
        gpu_layers=args.gpu_layers,
        gpu_device=args.gpu_device,
        kv_quant=not args.no_kv_quant,
        flash_attn=not args.no_flash_attn,
        model_filter=args.model_filter,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
