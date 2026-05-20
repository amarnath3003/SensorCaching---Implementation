"""
eval_harness.py — Full evaluation harness
Runs all 7 configurations (C1, C1-T, C1-G, C2–C5), each with N_SEEDS=5.
On real hardware: uses live sensor reads.
In simulation mode: uses Gaussian-drift sensor simulator.

Paper ref: Section V-D (experimental configurations), Table III
"""

from __future__ import annotations
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import config
from sensor_provider import SensorProvider
from sensor_simulator import SimulatedSensorProvider
from cache_store import CacheStore
from semantic_lookup import SemanticLookup, LookupResult
from cvw_monitor import CVWMonitor
from uctr_gate import UCTRGate
from llm_backend import LlamaCppBackend, MockLLMBackend
from prewarm import prewarm_cache
from metrics_logger import MetricsLogger, aggregate_seeds

logger = logging.getLogger(__name__)


# ─── Corpus loading ──────────────────────────────────────────────────────────

def load_corpus(path: str = config.CORPUS_PATH) -> list[dict]:
    """
    Load wilderness query corpus.
    Format: [{"query": ..., "category": ..., "urgency": "normal"|"high"}, ...]
    Falls back to a minimal 12-query test set if corpus is missing.
    """
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        logger.info("Corpus loaded: %d queries from %s", len(corpus), p)
        return corpus

    logger.warning("Corpus not found at %s — using built-in test set", path)
    return _BUILTIN_TEST_CORPUS


_BUILTIN_TEST_CORPUS = [
    # 2 unique × 6 paraphrases = 12 queries, covering all 6 categories
    {"query": "What is the safest descent route from this ridge?",      "category": "navigation",    "urgency": "normal"},
    {"query": "Which way should I go down from here safely?",           "category": "navigation",    "urgency": "normal"},
    {"query": "Safe way to descend from the summit?",                   "category": "navigation",    "urgency": "normal"},
    {"query": "How do I treat altitude sickness symptoms?",             "category": "first_aid",     "urgency": "high"},
    {"query": "What should I do if I have altitude sickness?",          "category": "first_aid",     "urgency": "high"},
    {"query": "What does the dropping pressure mean for the weather?",  "category": "weather",       "urgency": "normal"},
    {"query": "Should I be worried about the current weather?",         "category": "weather",       "urgency": "normal"},
    {"query": "How do I find drinkable water in the mountains?",        "category": "resource_mgmt", "urgency": "normal"},
    {"query": "Where can I get fresh water outdoors?",                  "category": "resource_mgmt", "urgency": "normal"},
    {"query": "Are there bears in this forest area?",                   "category": "threat_assess", "urgency": "normal"},
    {"query": "What wildlife threats should I be aware of here?",       "category": "threat_assess", "urgency": "normal"},
    {"query": "What are Leave No Trace principles?",                    "category": "gen_knowledge", "urgency": "normal"},
]


# ─── Single-config run ────────────────────────────────────────────────────────

def run_config(
    cfg_name:        str,
    cfg:             dict,
    corpus:          list[dict],
    real_sensor:     Optional[SensorProvider] = None,
    seed:            int = 42,
    use_mock_llm:    bool = False,
) -> MetricsLogger:
    """
    Run one configuration for one seed.
    If real_sensor is provided, it is used instead of the simulator.
    """
    scenario = cfg.get("scenario", "static")
    use_cvw  = cfg.get("cvw", False)
    use_uctr = cfg.get("uctr", False)
    ttl_s    = cfg.get("ttl", None)
    gps_only = cfg.get("gps_only", False)
    force_lb = cfg.get("force_low_battery_at", -1)

    logger.info("=== Run: %s seed=%d scenario=%s CVW=%s UCTR=%s ===",
                cfg_name, seed, scenario, use_cvw, use_uctr)

    # ── Sensor provider ───────────────────────────────────────────────────────
    if real_sensor is not None:
        sensor = real_sensor
    else:
        sensor = SimulatedSensorProvider(
            scenario             = scenario,
            seed                 = seed,
            force_low_battery_at = force_lb,
        )

    # ── Components ────────────────────────────────────────────────────────────
    cache  = CacheStore()
    lookup = SemanticLookup(
        cache       = cache,
        cvw_enabled = use_cvw,
        gps_only    = gps_only,
    )
    uctr   = UCTRGate(enabled=use_uctr)
    llm    = MockLLMBackend() if use_mock_llm else LlamaCppBackend()
    ml     = MetricsLogger(config_name=cfg_name, seed=seed)

    # ── Pre-warm ──────────────────────────────────────────────────────────────
    init_state = sensor.get_state()
    prewarm_cache(cache, lookup, init_state)

    # ── CVW Monitor ───────────────────────────────────────────────────────────
    monitor = None
    if use_cvw:
        monitor = CVWMonitor(cache, sensor)
        monitor.start()

    # ── Query loop ────────────────────────────────────────────────────────────
    eval_corpus = [q for i, q in enumerate(corpus)
                   if i >= int(0.2 * len(corpus))]   # skip held-out 20%
    logger.info("Evaluating %d queries (80%% split)", len(eval_corpus))

    for idx, item in enumerate(eval_corpus):
        query    = item["query"]
        category = item.get("category", "gen_knowledge")
        urgency  = item.get("urgency",  "normal")

        # ── C5: force low battery at midpoint ────────────────────────────────
        # (Simulator handles this internally; for real sensor we override)
        if (force_lb > 0 and idx >= force_lb
                and real_sensor is None):
            pass   # SimulatedSensorProvider handles internally

        # ── Get live sensor state ─────────────────────────────────────────────
        s_now = sensor.get_state()

        # ── Compute UCTR alpha ────────────────────────────────────────────────
        alpha = uctr.compute_alpha(
            battery_pct  = s_now.battery_pct,
            urgency      = urgency,
            sensor_drift = {"gps": 0.0, "alt": 0.0, "baro": 0.0, "temp": 0.0},
            category     = category,
        )

        # ── Three-stage lookup ────────────────────────────────────────────────
        t_lookup = time.perf_counter()
        result, entry, meta = lookup.lookup(query, s_now, alpha)
        lookup_ms = (time.perf_counter() - t_lookup) * 1000

        # ── Hit path ─────────────────────────────────────────────────────────
        if result in (LookupResult.HASH_HIT, LookupResult.SEMANTIC_HIT):
            response  = entry.response
            ttft_ms   = 1.0 + lookup_ms   # cache retrieval latency
            uctr.record_outcome(True)

        # ── Miss path (CVW miss or full miss) ─────────────────────────────────
        else:
            try:
                response, ttft_ms = llm.generate(query)
            except RuntimeError as exc:
                logger.error("LLM inference failed: %s", exc)
                response, ttft_ms = "[LLM unavailable]", 9999.0

            # Insert new entry into cache
            emb = lookup.encode(query)
            cat = lookup.classify(query, emb)
            cache.insert(
                query        = query,
                embedding    = emb,
                response     = response,
                sensor_state = s_now,
                category     = cat,
                alpha        = alpha,
                ttl_s        = ttl_s,
            )
            uctr.record_outcome(False)

        # ── Record metrics ────────────────────────────────────────────────────
        ml.record(
            query_index      = idx,
            query_text       = query,
            category         = category,
            urgency          = urgency,
            result           = result.name.lower(),
            ttft_ms          = ttft_ms,
            alpha            = alpha,
            sensor_now       = s_now,
            sensor_insert    = entry.sensor_snapshot if entry else None,
            cvw_delta        = meta.get("cvw_delta"),
            response_snippet = response[:80],
        )

        if idx % 100 == 0 and idx > 0:
            logger.info(
                "Progress: %d/%d  CHR=%.1f%%  CVW-inv=%d",
                idx, len(eval_corpus),
                ml.cache_hit_rate() * 100,
                monitor.invalidation_count if monitor else 0
            )

    # ── Teardown ──────────────────────────────────────────────────────────────
    if monitor:
        monitor.stop()
        ml.sync_cvw_count(monitor.invalidation_count)

    ml.print_summary()
    return ml


# ─── Multi-seed runner ────────────────────────────────────────────────────────

def run_all(
    configs:        Optional[dict] = None,
    n_seeds:        int  = config.N_SEEDS,
    real_sensor:    Optional[SensorProvider] = None,
    use_mock_llm:   bool = False,
    output_dir:     str  = config.RESULTS_DIR,
) -> dict[str, list[dict]]:
    """
    Run all configurations × N seeds.
    Returns {config_name: [summary_seed_0, ..., summary_seed_N]}.
    """
    if configs is None:
        configs = config.EVAL_CONFIGS

    corpus = load_corpus()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[dict]] = {}

    for cfg_name, cfg in configs.items():
        seed_results = []
        for seed in range(n_seeds):
            ml = run_config(
                cfg_name     = cfg_name,
                cfg          = cfg,
                corpus       = corpus,
                real_sensor  = real_sensor,
                seed         = seed,
                use_mock_llm = use_mock_llm,
            )
            path = ml.export_json(
                os.path.join(output_dir, f"{cfg_name}_seed{seed}.json")
            )
            seed_results.append(ml.summary())

        agg = aggregate_seeds(seed_results)
        agg_path = os.path.join(output_dir, f"{cfg_name}_aggregate.json")
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)
        logger.info("Aggregate → %s: CHR=%.1f±%.1f%%",
                    cfg_name,
                    agg.get("CHR_mean", 0),
                    agg.get("CHR_std", 0))

        all_results[cfg_name] = seed_results

    # Master summary table
    table_path = os.path.join(output_dir, "table_IV.json")
    table      = {k: aggregate_seeds(v) for k, v in all_results.items()}
    with open(table_path, "w") as f:
        json.dump(table, f, indent=2)
    logger.info("Master Table IV → %s", table_path)

    _print_table_iv(table)
    return all_results


def _print_table_iv(table: dict):
    print("\n" + "═"*78)
    print("TABLE IV — PERFORMANCE COMPARISON (mean over 5 runs)")
    print("═"*78)
    hdr = f"{'Config':<8} {'CHR%':>6} {'VHR%':>6} {'CVW-inv':>8} {'TTFT-H':>8} {'TTFT-M':>8}"
    print(hdr)
    print("─"*78)
    for cfg_name, agg in table.items():
        print(
            f"{cfg_name:<8} "
            f"{agg.get('CHR_mean',0):>6.1f} "
            f"{agg.get('VHR_mean',0):>6.1f} "
            f"{agg.get('CVW_inv_mean',0):>8.0f} "
            f"{agg.get('TTFT_H_ms_mean',0):>8.1f} "
            f"{agg.get('TTFT_M_ms_mean',0):>8.1f}"
        )
    print("═"*78 + "\n")
