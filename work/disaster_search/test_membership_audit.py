import io
import unittest
from unittest.mock import patch

from audit_xbd_membership import MAX_READ, RangeBudget, VerifiedRangeFile, sample_id


class Response(io.BytesIO):
    def __init__(self, value, status=206, content_range="bytes 5-8/20"):
        super().__init__(value)
        self.status = status
        self.headers = {"Content-Range": content_range}


class MetadataAdmissionChecks(unittest.TestCase):
    def test_partial_read_and_seek_are_bounded(self):
        budget = RangeBudget(100)
        with VerifiedRangeFile("https://example.invalid/data", 20, budget) as reader:
            reader.seek(5)
            with patch("audit_xbd_membership.urlopen", return_value=Response(b"abcd")):
                self.assertEqual(reader.read(4), b"abcd")
            self.assertEqual(reader.tell(), 9)
            self.assertEqual(budget.received, 4)
            self.assertEqual(reader.read_receipts[0]["offset"], 5)

    def test_non_range_response_is_rejected(self):
        with VerifiedRangeFile("https://example.invalid/data", 20, RangeBudget()) as reader:
            reader.seek(5)
            with patch("audit_xbd_membership.urlopen", return_value=Response(b"abcd", status=200)):
                with self.assertRaises(ValueError):
                    reader.read(4)

    def test_wrong_byte_range_is_rejected(self):
        with VerifiedRangeFile("https://example.invalid/data", 20, RangeBudget()) as reader:
            reader.seek(5)
            with patch("audit_xbd_membership.urlopen", return_value=Response(b"abcd", content_range="bytes 0-3/20")):
                with self.assertRaises(ValueError):
                    reader.read(4)

    def test_bulk_read_and_out_of_bounds_seek_are_rejected(self):
        with VerifiedRangeFile("https://example.invalid/data", MAX_READ + 1, RangeBudget()) as reader:
            with self.assertRaises(ValueError):
                reader.read()
            with self.assertRaises(ValueError):
                reader.seek(-1)

    def test_network_budget_cannot_be_overreserved(self):
        budget = RangeBudget(10)
        budget.reserve(6)
        with self.assertRaises(ValueError):
            budget.reserve(5)
        self.assertEqual(budget.reserved, 6)

    def test_identity_requires_author_split_and_canonical_name(self):
        self.assertEqual(sample_id("hold/images/hurricane-michael_00000108", "hold"), "hurricane-michael_00000108")
        for value in ("test/images/hurricane-michael_00000108", "hold/images/a.png", "hold/images/../secret"):
            with self.assertRaises(ValueError):
                sample_id(value, "hold")


if __name__ == "__main__":
    unittest.main()
