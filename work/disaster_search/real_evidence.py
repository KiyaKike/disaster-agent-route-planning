"""Keep correlated quality variants as one image evidence group.

This registry selects a representative observation. It does not infer likelihoods
or implement an independence assumption. No evaluator labels enter this module.
"""
from copy import deepcopy


class ImageEvidenceRegistry:
    MODE_RANK = {"blurred": 0, "native": 1}

    def __init__(self):
        self._seen = set()
        self._groups = {}

    def add(self, observation):
        sid, cell, mode = (observation[k] for k in ("sample_id", "cell_id", "mode"))
        group = f"{sid}:{cell}"
        oid = f"{group}:{mode}"
        if observation["evidence_group_id"] != group or observation["observation_id"] != oid:
            raise ValueError("Noncanonical evidence identity")
        if mode not in self.MODE_RANK:
            raise ValueError("Unknown observation mode")
        if oid in self._seen:
            raise ValueError("Duplicate observation ID")
        self._seen.add(oid)
        # Whitelist fields to prevent evaluator-only object IDs leaking through.
        selected = {k: deepcopy(observation[k]) for k in
                    ("sample_id", "cell_id", "mode", "evidence_group_id", "observation_id", "declared_target",
                     "uncalibrated_severe_score")}
        previous = self._groups.get(group)
        replace = previous is None or self.MODE_RANK[mode] > self.MODE_RANK[previous["mode"]]
        if replace:
            self._groups[group] = selected
        return replace

    def representatives(self):
        return deepcopy([self._groups[k] for k in sorted(self._groups)])
