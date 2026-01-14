import json
from typing import Any

from ._config import config, CustomDict, get_package_root


def _get_paths():
    """Get dataset and movie paths from package root."""
    root = get_package_root()
    path_dataset = root / config.path.dataset
    path_movies = root / config.path.src
    return path_dataset, path_movies


class Info:
    def __init__(self, path):
        self.path = path
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.data, f)
        for id in self.data:
            self.data[id] = CustomDict(self.data[id])
        self._set_id2path()

    def _set_id2path(self):
        path_dataset, _ = _get_paths()
        id2path = {}
        for id_piano, info in self.data.items():
            id_orig = info["original"]
            title = info["title"]
            if id_orig not in id2path:
                id2path[id_orig] = {
                    "raw": path_dataset / "raw" / title / f"{id_orig}.wav",
                    "synced": {
                        "wav": path_dataset / "synced" / title / f"{id_orig}.wav",
                        "midi": path_dataset / "synced" / title / f"{id_orig}.mid",
                    },
                    "array": path_dataset / "array" / title / f"{id_orig}.npy",
                }
            id2path[id_piano] = {
                "raw": path_dataset / "raw" / title / "piano" / f"{id_piano}.wav",
                "synced": {
                    "wav": path_dataset / "synced" / title / "piano" / f"{id_piano}.wav",
                    "midi": path_dataset / "synced" / title / "piano" / f"{id_piano}.mid",
                },
                "array": path_dataset / "array" / title / "piano" / f"{id_piano}.npz",
            }
        self._id2path = CustomDict(id2path)

    def __getitem__(self, id: str):
        return self.data[id]

    def set(self, id: str, key: str, value: Any, save: bool = True):
        if id not in self.data:
            self.data[id] = {}
        self.data[id][key] = value
        if save:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)

    def update(self, id: str, values: dict, save: bool = True):
        if id not in self.data:
            self.data[id] = {}
        self.data[id].update(values)
        if save:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)

    def export(self):
        _, path_movies = _get_paths()
        movies = {}
        for id, info in self.data.items():
            if not info["include_dataset"]:
                continue

            title = info["title"]
            if title not in movies:
                movies[title] = {
                    "original": info["original"],
                    "pianos": []
                }
            movies[title]["pianos"].append(id)

        with open(path_movies, "w", encoding="utf-8") as f:
            json.dump(movies, f, indent=2, ensure_ascii=False)

    def piano2orig(self, id: str):
        return self[id].original

    def is_train(self, id: str):
        return (self[id].split == "train")

    def is_test(self, id: str):
        return (self[id].split == "test")

    def id2path(self, id: str, orig: bool = False):
        if orig:
            return self._id2path[self.piano2orig(id)]
        else:
            return self._id2path[id]

    def get_ids(self, split: str, orig: bool = False):
        ids = [id for id, info in self.data.items() if info["split"] == split]
        if orig:
            ids = list(set([self.piano2orig(id) for id in ids]))
        return ids


# Initialize info at module load time
_root = get_package_root()
info = Info(_root / config.path.info)
