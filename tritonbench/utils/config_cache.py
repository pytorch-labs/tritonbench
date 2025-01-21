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
        # os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        self._load_cache()

    def _tuple_to_key(self, t):
        """Convert a tuple to a JSON-compatible string key."""
        return f"{json.dumps(tuple(t))}"

    def _key_to_tuple(self, key):
        """Convert a JSON-compatible string key back to a tuple."""
        if key.startswith("key_"):
            return tuple(json.loads(key[4:]))
        return key
    
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
                "configs": {}
            }
            self._save_cache()

    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def get_key(self, B, M, seq_lengths, MAX_SEQLEN):
        sorted_lengths = tuple(sorted(seq_lengths))
        
        # Return tuple key
        key_tuple = (B, M, sorted_lengths, MAX_SEQLEN)
        return self._tuple_to_key(key_tuple)

    def get_config(self, B, M, seq_lengths, MAX_SEQLEN):
        key = self.get_key(B, M, seq_lengths, MAX_SEQLEN)
        return self.cache["configs"].get(key)

    #ignore storing_configs
    def store_config(self, B, M, seq_lengths, MAX_SEQLEN, config, perf = 0.0):
        key = self.get_key(B, M, seq_lengths, MAX_SEQLEN)
        current = self.cache["configs"].get(key)
        
        # print(f"Storing config for key {key}")  # Debug print
        # print(f"Config: {config}")
        if current is None or perf < current["perf"]:
            self.cache["configs"][key] = {
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
            self.cache["metadata"]["last_updated"] = datetime.datetime.now().isoformat()
            self._save_cache()