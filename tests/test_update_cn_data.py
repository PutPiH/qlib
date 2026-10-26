"""Offline checks for price continuity, suspension gaps, and safe binary updates."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dump_bin import DumpDataAll
from update_cn_data import DoltHub, FIELDS, INDEX_MAP, apply_update, index_intervals, normalize_increment, prepare, publish, read_bins


def prices(symbol, dates, closes, adjusted):
    return pd.DataFrame({"symbol": symbol, "tradedate": dates, "open": closes, "high": closes,
                         "low": closes, "close": closes, "adjclose": adjusted,
                         "volume": 100.0, "amount": 1000.0})


class TestIncrementalData(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.to_datetime(["2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13", "2026-10-14"])
        self.raw_old = prices("SH600000", self.calendar[:2], [10.0, 11.0], [100.0, 110.0])
        self.old = normalize_increment(self.raw_old, self.calendar).set_index("tradedate")
        self.anchor = self.raw_old.iloc[-1]

    def test_split_and_suspension_preserve_price_scale(self):
        raw = prices("SH600000", ["2026-10-14"], [6.0], [120.0])
        result = normalize_increment(raw, self.calendar, self.old, self.anchor).set_index("tradedate")
        self.assertEqual(result.index.tolist(), self.calendar[2:].tolist())
        self.assertTrue(result["close"].iloc[:2].isna().all())
        np.testing.assert_allclose(result["factor"].iloc[:2], [0.1, 0.1])
        self.assertAlmostEqual(result["close"].iloc[-1], 1.2)
        self.assertAlmostEqual(result["factor"].iloc[-1], 0.2)
        self.assertAlmostEqual(result["volume"].iloc[-1], 500)
        self.assertEqual(result["amount"].iloc[-1], 1000)
        self.assertAlmostEqual(result["vwap"].iloc[-1], 20)
        self.assertAlmostEqual(result["change"].iloc[-1], 6 / 11 - 1)

    def test_new_stock_starts_at_one(self):
        raw = prices("SZ001001", self.calendar[2:], [20.0, 21.0, 22.0], [200.0, 210.0, 220.0])
        result = normalize_increment(raw, self.calendar)
        np.testing.assert_allclose(result["close"], [1, 1.05, 1.1])
        self.assertTrue(pd.isna(result["change"].iloc[0]))

    def test_reject_changed_history_and_duplicate_dates(self):
        raw = prices("SH600000", ["2026-10-14"], [6.0], [120.0])
        changed = self.anchor.copy()
        changed["close"] = 12
        with self.assertRaisesRegex(ValueError, "Historical source changed"):
            normalize_increment(raw, self.calendar, self.old, changed)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            normalize_increment(pd.concat([raw, raw]), self.calendar, self.old, self.anchor)

    def test_stage_append_preserves_old_bytes_and_new_stock(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "old_csv"
            source.mkdir()
            self.old.reset_index().to_csv(source / "SH600000.csv", index=False)
            root = tmp / "live"
            DumpDataAll(data_path=str(source), qlib_dir=str(root), date_field_name="tradedate",
                        exclude_fields="tradedate,symbol", max_workers=1).dump()
            work = tmp / "work"
            normalized = work / "normalized"
            normalized.mkdir(parents=True)
            update = normalize_increment(prices("SH600000", ["2026-10-14"], [6.0], [120.0]),
                                         self.calendar, self.old, self.anchor)
            update.to_csv(normalized / "SH600000.csv", index=False)
            normalize_increment(prices("SZ001001", self.calendar[2:], [20., 21., 22.], [200., 210., 220.]),
                                self.calendar).to_csv(normalized / "SZ001001.csv", index=False)
            original = (root / "features/sh600000/close.day.bin").read_bytes()
            stage = apply_update(root, work, (normalized, self.calendar, self.calendar, {}), "test-commit")
            self.assertEqual((root / "features/sh600000/close.day.bin").read_bytes(), original)
            self.assertEqual((root / "calendars/day.txt").read_text().splitlines()[-1], "2026-10-09")
            actual = read_bins(stage, "SH600000", self.calendar)
            np.testing.assert_allclose(actual["close"], [1., 1.1, np.nan, np.nan, 1.2], equal_nan=True)
            new = read_bins(stage, "SZ001001", self.calendar)
            self.assertEqual(new.index[0], self.calendar[2])
            self.assertEqual(set(FIELDS), set(new.columns))

    def test_repeated_index_membership_is_preserved(self):
        weights = pd.DataFrame({"trade_date": ["2026-10-08", "2026-10-09", "2026-10-12"],
                                "stock_code": ["600000.SH", "000001.SZ", "600000.SH"]})
        result = index_intervals(weights, self.calendar[-1])
        self.assertIn("SH600000\t2026-10-08\t2026-10-08\n", result)
        self.assertIn("SH600000\t2026-10-12\t2026-10-14\n", result)

    def test_up_to_date_is_noop(self):
        class UpToDateDolt:
            def sql(inner, query):
                if "MAX(tradedate)" in query:
                    return pd.DataFrame({"date": ["2026-10-09"]})
                if "ts_trade_day_calendar" in query:
                    return pd.DataFrame({"date": self.calendar})
                return pd.DataFrame(columns=["symbol", "tradedate"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "calendars").mkdir()
            (root / "instruments").mkdir()
            (root / "calendars/day.txt").write_text("2026-10-08\n2026-10-09\n")
            (root / "instruments/all.txt").write_text("SH600000\t2026-10-08\t2026-10-09\n")
            self.assertIsNone(prepare(UpToDateDolt(), root, root))

    def test_public_api_paginates_and_pins_snapshot(self):
        database = DoltHub.__new__(DoltHub)
        database.commit = "a" * 32
        database.session = Mock()
        schema = [{"columnName": "symbol"}]
        first, last = Mock(), Mock()
        first.json.return_value = {"query_execution_status": "RowLimit", "schema": schema,
                                   "rows": [{"symbol": str(i)} for i in range(1000)]}
        last.json.return_value = {"query_execution_status": "Success", "schema": schema,
                                  "rows": [{"symbol": "last"}]}
        database.session.get.side_effect = [first, last]
        result = database.sql("SELECT symbol FROM final_a_stock_eod_price ORDER BY symbol")
        self.assertEqual(len(result), 1001)
        query = database.session.get.call_args.kwargs["params"]["q"]
        self.assertIn("AS OF '" + database.commit + "'", query)
        self.assertIn("OFFSET 1000", query)

    def test_public_api_rejects_query_errors(self):
        database = DoltHub.__new__(DoltHub)
        database.commit = None
        database.session = Mock()
        database.session.get.return_value.json.return_value = {
            "query_execution_status": "Error", "query_execution_message": "query timed out"}
        with self.assertRaisesRegex(ValueError, "query timed out"):
            database.sql("SELECT 1")

    def test_prepare_prices_and_all_index_updates(self):
        raw = prices("SH600000", self.calendar[2:], [5.5, 5.75, 6.0], [110., 115., 120.])

        class SnapshotDolt:
            def sql(inner, query):
                if "MAX(tradedate)" in query:
                    return pd.DataFrame({"date": ["2026-10-14"]})
                if "ts_trade_day_calendar" in query:
                    return pd.DataFrame({"date": self.calendar})
                if "WHERE tradedate >" in query:
                    return raw.copy()
                if "WHERE (symbol,tradedate)" in query:
                    return self.raw_old.tail(1).copy()
                if "SELECT trade_date FROM (" in query:
                    return pd.DataFrame({"trade_date": ["2026-10-09"]})
                if "SELECT trade_date,stock_code" in query:
                    return pd.DataFrame({"trade_date": ["2026-10-09", "2026-10-14"],
                                         "stock_code": ["600000.SH", "000001.SZ"]})
                raise AssertionError(f"Unexpected query: {query}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "calendars").mkdir()
            (root / "instruments").mkdir()
            features = root / "features/sh600000"
            features.mkdir(parents=True)
            for field in FIELDS:
                np.r_[0, self.old[field].to_numpy()].astype("<f4").tofile(features / f"{field}.day.bin")
            (root / "calendars/day.txt").write_text("2026-10-08\n2026-10-09\n")
            (root / "instruments/all.txt").write_text("SH600000\t2026-10-08\t2026-10-09\n")
            for name in INDEX_MAP:
                (root / "instruments" / f"{name}.txt").write_text("SH600000\t2026-10-08\t2026-10-09\n")
            normalized, calendar, _, indices = prepare(SnapshotDolt(), root, root)
            result = pd.read_csv(normalized / "SH600000.csv")
            np.testing.assert_allclose(result["close"], [1.1, 1.15, 1.2])
            self.assertTrue(calendar.equals(self.calendar))
            self.assertEqual(set(indices), set(INDEX_MAP))
            for text in indices.values():
                self.assertIn("SH600000\t2026-10-08\t2026-10-08\n", text)
                self.assertIn("SH600000\t2026-10-09\t2026-10-13\n", text)
                self.assertIn("SZ000001\t2026-10-14\t2026-10-14\n", text)

    def test_failed_directory_replacement_restores_live_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, stage = Path(tmp) / "live", Path(tmp) / "stage"
            root.mkdir()
            stage.mkdir()
            previous = root.with_name("live.backup-20261001-000000")
            previous.mkdir()
            (previous / "marker").write_text("previous backup")
            (root / "marker").write_text("old data")
            rename = Path.rename

            def fail_stage(path, target):
                if path == stage:
                    raise OSError("simulated replacement failure")
                return rename(path, target)

            with patch.object(Path, "rename", fail_stage):
                with self.assertRaisesRegex(OSError, "simulated"):
                    publish(root, stage)
            self.assertEqual((root / "marker").read_text(), "old data")
            self.assertTrue(stage.exists())
            self.assertEqual((previous / "marker").read_text(), "previous backup")

    def test_successful_replacement_keeps_only_latest_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, stage = Path(tmp) / "live", Path(tmp) / "stage"
            root.mkdir()
            stage.mkdir()
            (root / "marker").write_text("old data")
            (stage / "marker").write_text("new data")
            previous = root.with_name("live.backup-20261001-000000")
            previous.mkdir()
            (previous / "marker").write_text("obsolete")
            unrelated = root.with_name("live.backup-manual")
            unrelated.mkdir()
            other_dataset = root.with_name("other.backup-20261001-000000")
            other_dataset.mkdir()
            link = root.with_name("live.backup-20261002-000000")
            link.symlink_to(unrelated, target_is_directory=True)
            backup = publish(root, stage)
            self.assertEqual((root / "marker").read_text(), "new data")
            self.assertEqual((backup / "marker").read_text(), "old data")
            self.assertFalse(previous.exists())
            self.assertTrue(unrelated.exists())
            self.assertTrue(other_dataset.exists())
            self.assertTrue(link.is_symlink())

    def test_backup_cleanup_failure_does_not_undo_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, stage = Path(tmp) / "live", Path(tmp) / "stage"
            root.mkdir()
            stage.mkdir()
            (stage / "marker").write_text("new data")
            previous = root.with_name("live.backup-20261001-000000")
            previous.mkdir()
            with patch("update_cn_data.shutil.rmtree", side_effect=PermissionError("denied")):
                backup = publish(root, stage)
            self.assertTrue(backup.is_dir())
            self.assertTrue(previous.is_dir())
            self.assertEqual((root / "marker").read_text(), "new data")


if __name__ == "__main__":
    unittest.main()
