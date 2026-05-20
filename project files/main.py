"""
main.py — SensorAware-Cache entry point
Usage examples:

  # Full simulated eval (all configs, 5 seeds) — fastest path to paper metrics
  python main.py --mode sim --configs C1,C2,C3,C4,C5 --seeds 5

  # RPi5 real-sensor single run
  PLATFORM=rpi python main.py --mode real --config C4 --seeds 1

  # Android real-sensor single run
  PLATFORM=android python main.py --mode real --config C4 --seeds 1

  # Quick smoke-test with mock LLM (no GPU needed)
  python main.py --mode sim --mock-llm --configs C1,C4 --seeds 1

  # Interactive REPL (real-time query answering on device)
  python main.py --mode interactive
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

import config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SensorAware-Cache evaluation harness",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["sim", "real", "interactive"], default="sim",
        help="sim=Gaussian simulator | real=live hardware | interactive=REPL",
    )
    p.add_argument(
        "--configs", default="C1,C2,C3,C4,C5",
        help="Comma-separated config names (default: C1,C2,C3,C4,C5)",
    )
    p.add_argument("--seeds",    type=int,  default=5)
    p.add_argument("--mock-llm", action="store_true",
                   help="Use mock LLM backend (no llama.cpp required)")
    p.add_argument("--platform", default=None,
                   help="Override PLATFORM env var: rpi|android|sim")
    p.add_argument("--model",    default=config.MODEL_PATH)
    p.add_argument("--output",   default=config.RESULTS_DIR)
    p.add_argument("--verbose",  action="store_true")
    return p


def run_simulation(args: argparse.Namespace):
    """Simulation mode: Gaussian-drift sensor, full eval harness."""
    from eval_harness import run_all

    cfg_names = [c.strip() for c in args.configs.split(",")]
    selected  = {k: v for k, v in config.EVAL_CONFIGS.items() if k in cfg_names}
    if not selected:
        logger.error("No valid configs selected from: %s", args.configs)
        sys.exit(1)

    logger.info("Simulation eval: configs=%s  seeds=%d  mock_llm=%s",
                list(selected.keys()), args.seeds, args.mock_llm)

    run_all(
        configs      = selected,
        n_seeds      = args.seeds,
        real_sensor  = None,
        use_mock_llm = args.mock_llm,
        output_dir   = args.output,
    )


def run_real(args: argparse.Namespace):
    """Real-device mode: live sensors, full eval harness."""
    platform = args.platform or os.environ.get("PLATFORM", "rpi")
    os.environ["PLATFORM"] = platform

    from sensor_provider import get_provider
    from eval_harness import run_all

    logger.info("Initialising real sensor provider for platform=%s", platform)
    sensor = get_provider(platform)

    if not sensor.is_healthy():
        logger.error(
            "Sensor health check failed.  "
            "Check hardware connections / Termux:API permissions."
        )
        sys.exit(1)

    logger.info("Sensor OK: %s", sensor.get_state())

    cfg_names = [c.strip() for c in args.configs.split(",")]
    selected  = {k: v for k, v in config.EVAL_CONFIGS.items() if k in cfg_names}

    run_all(
        configs      = selected,
        n_seeds      = args.seeds,
        real_sensor  = sensor,
        use_mock_llm = args.mock_llm,
        output_dir   = args.output,
    )


def run_interactive(args: argparse.Namespace):
    """
    Interactive REPL — demonstrates SensorAware-Cache in real time.
    Queries the live LLM via cache, shows cache hit/miss + TTFT.
    """
    platform = args.platform or os.environ.get("PLATFORM", "rpi")
    os.environ["PLATFORM"] = platform

    from sensor_provider import get_provider
    from cache_store import CacheStore
    from semantic_lookup import SemanticLookup, LookupResult
    from cvw_monitor import CVWMonitor
    from uctr_gate import UCTRGate
    from llm_backend import LlamaCppBackend, MockLLMBackend
    from prewarm import prewarm_cache

    logger.info("Interactive mode | platform=%s", platform)

    sensor = get_provider(platform)
    cache  = CacheStore()
    lookup = SemanticLookup(cache, cvw_enabled=True)
    uctr   = UCTRGate(enabled=True)
    llm    = MockLLMBackend() if args.mock_llm else LlamaCppBackend()

    init_state = sensor.get_state()
    n_warm = prewarm_cache(cache, lookup, init_state)
    logger.info("Pre-warmed with %d entries", n_warm)

    monitor = CVWMonitor(cache, sensor)
    monitor.start()

    print("\n" + "─"*60)
    print(" SensorAware-Cache interactive demo")
    print(" Type your query. 'quit' to exit. 'status' for cache info.")
    print("─"*60)

    try:
        while True:
            try:
                query = input("\nQuery> ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if query.lower() in ("quit", "exit", "q"):
                break

            if query.lower() == "status":
                s = sensor.get_state()
                print(f"  Sensor: {s}")
                print(f"  Cache size:   {cache.size()}")
                print(f"  CVW inv:      {monitor.invalidation_count}")
                continue

            if not query:
                continue

            s_now = sensor.get_state()
            alpha = uctr.compute_alpha(
                battery_pct  = s_now.battery_pct,
                urgency      = "normal",
                sensor_drift = {"gps":0,"alt":0,"baro":0,"temp":0},
                category     = "gen_knowledge",
            )

            t0 = time.perf_counter()
            result, entry, meta = lookup.lookup(query, s_now, alpha)

            if result in (LookupResult.HASH_HIT, LookupResult.SEMANTIC_HIT):
                response = entry.response
                ttft_ms  = (time.perf_counter() - t0) * 1000
                tag      = "✅ CACHE HIT"
                uctr.record_outcome(True)
            else:
                response, ttft_ms = llm.generate(query)
                emb = lookup.encode(query)
                cat = lookup.classify(query, emb)
                cache.insert(query, emb, response, s_now, cat, alpha)
                tag = "🔴 CACHE MISS"
                uctr.record_outcome(False)

            sim_str = f"sim={meta.get('sim_score', 'N/A'):.3f}" \
                if meta.get("sim_score") else ""
            print(f"\n  [{tag}]  TTFT={ttft_ms:.1f}ms  α={alpha:.2f}  {sim_str}")
            print(f"\n{response[:400]}")
            if len(response) > 400:
                print("  ... [truncated]")

    finally:
        monitor.stop()
        print("\nExiting. CVW invalidations during session:", monitor.invalidation_count)


def main():
    parser  = build_parser()
    args    = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.model != config.MODEL_PATH:
        os.environ["LLAMA_MODEL_PATH"] = args.model

    if args.mode == "sim":
        run_simulation(args)
    elif args.mode == "real":
        run_real(args)
    elif args.mode == "interactive":
        run_interactive(args)


if __name__ == "__main__":
    main()
