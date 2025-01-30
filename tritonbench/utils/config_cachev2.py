import json
import os
import datetime
from pathlib import Path

class AutotuneCache:
    def __init__(self, cache_file="kernel_configs.json"):
        cache_dir = Path("./cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / cache_file
        print(f"Initializing cache at: {self.cache_file}")
        self._load_cache()
        self._build_index()  # Build in-memory index for fast lookups

    def _load_cache(self):
        try:
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.cache = {
                "metadata": {
                    "version": "1.0",
                    "last_updated": datetime.datetime.now().isoformat(),
                },
                "configs": []  # Array of configs
            }
            self._save_cache()

    def _build_index(self):
        """Build in-memory index for fast lookups"""
        self._index = {}
        for idx, config_entry in enumerate(self.cache["configs"]):
            key = json.dumps(config_entry["key"])
            self._index[key] = idx

    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
        self._build_index()

    def get_config(self, B, M, seq_lengths, MAX_SEQLEN):
        key = json.dumps([B, M, sorted(seq_lengths), MAX_SEQLEN])
        idx = self._index.get(key)
        if idx is not None:
            return self.cache["configs"][idx]["data"]
        return None

    def store_config(self, B, M, seq_lengths, MAX_SEQLEN, config, perf=0.0):
        key = [B, M, sorted(seq_lengths), MAX_SEQLEN]
        key_str = json.dumps(key)
        
        # Check
        idx = self._index.get(key_str)
        if idx is not None:
            if perf < self.cache["configs"][idx]["data"]["perf"]:
                self.cache["configs"][idx]["data"].update({
                    "config": config,
                    "perf": perf,
                    "timestamp": datetime.datetime.now().isoformat(),
                })
                self._save_cache()
            return

        # If not found, append 
        new_entry = {
            "key": key,
            "data": {
                "config": config,
                "perf": perf,
                "timestamp": datetime.datetime.now().isoformat(),
                "metadata": {
                    "B": B,
                    "M": M,
                    "max_seqlen": MAX_SEQLEN,
                    "seq_lengths_stats": {
                        "min": min(seq_lengths),
                        "max": max(seq_lengths),
                        "avg": sum(seq_lengths) / len(seq_lengths),
                        "num_sequences": len(seq_lengths)
                    }
                }
            }
        }
        self.cache["configs"].append(new_entry)
        self._index[key_str] = len(self.cache["configs"]) - 1
        self.cache["metadata"]["last_updated"] = datetime.datetime.now().isoformat()
        self._save_cache()