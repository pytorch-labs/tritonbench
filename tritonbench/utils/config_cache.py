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

    def get_key(self, M, MAX_SEQLEN):
        return f"{M}_{MAX_SEQLEN}"

    def get_config(self, M, MAX_SEQLEN):
        key = self.get_key(M, MAX_SEQLEN)
        return self.cache["configs"].get(key)

    #ignore storing_config
    #TODO: Implement storing config
    def store_config(self, M, MAX_SEQLEN, config, perf = 0.0):
        key = self.get_key(M, MAX_SEQLEN)
        current = self.cache["configs"].get(key)
        
        # print(f"Storing config for key {key}")  # Debug print
        # print(f"Config: {config}")
        if current is None or perf < current["perf"]:
            self.cache["configs"][key] = {
                "config": config,
                "perf": perf,
                "timestamp": datetime.datetime.now().isoformat()
            }
            self.cache["metadata"]["last_updated"] = datetime.datetime.now().isoformat()
            self._save_cache()