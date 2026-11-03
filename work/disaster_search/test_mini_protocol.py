import json
from pathlib import Path
import tempfile
import unittest

from build_xbd_mini import (DIAGNOSTIC_IDS, EVENTS, REVISION, complete_inventory,
                            create_plan, fetch_one, paths_for, select_samples)


def fake_inventory():
    return {event: [f"{event}_{i:08d}" for i in range(24)] for events in EVENTS.values() for event in events}


class MiniProtocolChecks(unittest.TestCase):
    def test_fixed_count_event_disjoint_and_diagnostic_exclusion(self):
        samples = select_samples(fake_inventory())
        self.assertEqual(len(samples), 72)
        groups = {s: {row["event"] for row in samples if row["split"] == s} for s in EVENTS}
        self.assertFalse(groups["calibration"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])
        self.assertFalse(groups["calibration"] & groups["validation"])
        self.assertFalse({row["sample_id"] for row in samples} & DIAGNOSTIC_IDS)
        for split in EVENTS:
            self.assertEqual(sum(s["split"] == split for s in samples), 24)

    def test_order_independent_selection_and_stable_seed(self):
        inventory = fake_inventory()
        reversed_inventory = {k: list(reversed(v)) for k, v in reversed(list(inventory.items()))}
        self.assertEqual(select_samples(inventory), select_samples(reversed_inventory))
        self.assertNotEqual(select_samples(inventory), select_samples(inventory, seed=999))

    def test_inventory_requires_four_complete_files(self):
        full = paths_for("example_00000000")
        partial = list(paths_for("example_00000001").values())[:-1]
        index = {"siblings": [{"rfilename": p} for p in list(full.values()) + partial]}
        self.assertEqual(complete_inventory(index), {"example": ["example_00000000"]})

    def test_no_substitution_for_insufficient_events(self):
        with self.assertRaises(ValueError):
            select_samples(fake_inventory(), count=100)
        with self.assertRaises(ValueError):
            select_samples(fake_inventory(), {"a": ["same"], "b": ["same"]})

    def test_freeze_receipt_and_tamper_detection(self):
        index = {"sha": REVISION, "siblings": [{"rfilename": p} for ids in fake_inventory().values()
                                               for sid in ids for p in paths_for(sid).values()]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "index.json"
            source.write_text(json.dumps(index))
            out = root / "mini"
            create_plan(source, out)
            first = (out / "protocol.json").read_bytes()
            create_plan(source, out)
            self.assertEqual((out / "protocol.json").read_bytes(), first)
            protocol = json.loads(first)
            protocol["seed"] = 1
            (out / "protocol.json").write_text(json.dumps(protocol))
            with self.assertRaises(ValueError):
                create_plan(source, out)

    def test_corrupt_cached_input_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = "train/images/a.png"
            (root / path).parent.mkdir(parents=True)
            (root / path).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                fetch_one(path, root, {path: {"sha256": "incorrect"}})


if __name__ == "__main__":
    unittest.main()
