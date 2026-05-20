# prewarm.py
# CVW-annotated pre-warming

"""
prewarm.py — CVW-annotated cache pre-warming
Loads domain-relevant query–response pairs at deployment time.
Each entry is tagged PRE_WARM=True and assigned appropriate CVW envelopes.

Paper ref: Section IV-E

Prewarm corpus format (prewarm_entries.json):
[
  {
    "query": "What is the safest descent route?",
    "response": "[Generic pre-warm] Descend along the most gradual slope...",
    "category": "navigation",
    "low_confidence": true
  },
  ...
]
"""

from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from sensor_provider import SensorState
from cache_store import CacheStore
from semantic_lookup import SemanticLookup
import config

logger = logging.getLogger(__name__)


# ─── Minimal pre-warm corpus (built-in fallback) ─────────────────────────────
# Used when prewarm_entries.json is absent.

_BUILTIN_PREWARM = [
    {
        "query": "What is the safest descent route from here?",
        "response": (
            "[PRE-WARM — refresh once GPS/baro are live] "
            "For safe descent: follow established trails, avoid steep scree, "
            "check for loose rock. Re-ask after reaching your destination for "
            "context-specific guidance."
        ),
        "category": "navigation",
        "low_confidence": True,
    },
    {
        "query": "What does the current weather mean for my safety?",
        "response": (
            "[PRE-WARM] Monitor pressure trends; a drop of >3 hPa/hour "
            "signals incoming bad weather. Seek shelter if thunder approaches. "
            "Re-ask for current conditions once connected to sensor data."
        ),
        "category": "weather",
        "low_confidence": True,
    },
    {
        "query": "How do I treat a blister in the wilderness?",
        "response": (
            "Clean the area. If intact, pad around the blister. "
            "If broken, clean with antiseptic, apply sterile dressing. "
            "Moleskin padding prevents further friction."
        ),
        "category": "first_aid",
        "low_confidence": False,
    },
    {
        "query": "How do I find and purify drinking water outdoors?",
        "response": (
            "Collect from running streams upstream of human activity. "
            "Boil for 1 minute (3 min above 2000 m) or use iodine/filter. "
            "Running water is generally safer than stagnant pools."
        ),
        "category": "resource_mgmt",
        "low_confidence": False,
    },
    {
        "query": "What are the Leave No Trace principles?",
        "response": (
            "1. Plan ahead. 2. Travel/camp on durable surfaces. "
            "3. Dispose of waste properly. 4. Leave what you find. "
            "5. Minimise fire impact. 6. Respect wildlife. "
            "7. Be considerate of other visitors."
        ),
        "category": "gen_knowledge",
        "low_confidence": False,
    },
    {
        "query": "What are signs of altitude sickness?",
        "response": (
            "Symptoms: headache, nausea, dizziness, fatigue, loss of appetite. "
            "Severe: confusion, ataxia, breathlessness at rest. "
            "Descend immediately if severe symptoms appear."
        ),
        "category": "first_aid",
        "low_confidence": False,
    },
    {
        "query": "How do I navigate without a GPS signal?",
        "response": (
            "Use map + compass: orient map to north, take bearing to landmark. "
            "Sun rises E, sets W; shadows point north (N hemisphere) midday. "
            "Follow watersheds downhill to populated areas."
        ),
        "category": "navigation",
        "low_confidence": False,
    },
    {
        "query": "How do I assess bear threat and stay safe?",
        "response": (
            "Make noise while hiking. Store food in bear canisters 200m from camp. "
            "If encountered: stand tall, speak calmly, back away slowly. "
            "Carry bear spray accessible at all times."
        ),
        "category": "threat_assess",
        "low_confidence": False,
    },
]


def load_prewarm_corpus(path: Optional[str] = None) -> list[dict]:
    """Load pre-warm entries from JSON file or fall back to built-in set."""
    p = Path(path or config.PREWARM_PATH)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            entries = json.load(f)
        logger.info("Loaded %d pre-warm entries from %s", len(entries), p)
        return entries
    else:
        logger.info("Prewarm corpus not found at %s — using built-in %d entries",
                    p, len(_BUILTIN_PREWARM))
        return _BUILTIN_PREWARM


def prewarm_cache(
    cache:        CacheStore,
    lookup:       SemanticLookup,
    sensor_state: SensorState,
    path:         Optional[str] = None,
) -> int:
    """
    Load pre-warm corpus and insert entries into the cache.
    Each entry gets PRE_WARM=True flag and a low-confidence indicator
    appended to state-sensitive categories.

    Returns: number of entries inserted.
    """
    corpus   = load_prewarm_corpus(path)
    inserted = 0
    t0       = time.time()

    for item in corpus:
        query    = item["query"]
        response = item["response"]
        category = item.get("category", "gen_knowledge")
        is_low   = item.get("low_confidence", False)

        # Append low-confidence marker for state-sensitive categories
        if is_low:
            response = (
                response + "\n\n[⚠ Pre-warm entry: context was not measured at "
                "your current location. Response may not reflect local conditions. "
                "Re-ask to refresh.]"
            )

        emb  = lookup.encode(query)
        cat  = lookup.classify(query, emb)   # verify / override category label
        if category != cat:
            logger.debug("Prewarm category override: declared=%s inferred=%s", category, cat)

        cache.insert(
            query        = query,
            embedding    = emb,
            response     = response,
            sensor_state = sensor_state,
            category     = category,
            alpha        = 1.0,
            pre_warm     = True,
        )
        inserted += 1

    elapsed = time.time() - t0
    logger.info(
        "Pre-warm complete: %d entries in %.1f s  (cache size=%d)",
        inserted, elapsed, cache.size()
    )
    return inserted
