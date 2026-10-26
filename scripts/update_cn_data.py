"""Incrementally import chenditc's public Dolt prices into an existing Qlib dataset."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timedelta
import fcntl
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from dump_bin import DumpDataUpdate


FIELDS = ("open", "high", "low", "close", "volume", "factor", "adjclose", "amount", "vwap", "change")
INDEX_MAP = {"csi300": "399300.SZ", "csi500": "000905.SH", "csi800": "000906.SH",
             "csi1000": "000852.SH", "csiall": "000985.SH"}
SYMBOL = re.compile(r"(?:SH|SZ|BJ)[0-9]{6}\Z")


def log(message):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


class DoltHub:
    """Read the public database at one immutable commit, in pages of at most 1000 rows."""

    URL = "https://www.dolthub.com/api/v1alpha1/chenditc/investment_data/master"

    def __init__(self, proxy=None):
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
        if proxy:
            self.session.proxies.update({"http": proxy, "https": proxy})
        self.commit = None
        self.commit = str(self.sql("SELECT DOLT_HASHOF('HEAD') AS hash").iloc[0, 0])
        if re.fullmatch(r"[0-9a-v]{32}", self.commit) is None:
            raise ValueError("Invalid Dolt commit hash")

    def sql(self, query):
        if self.commit:
            query = re.sub(r"\b(FROM|JOIN)\s+(final_a_stock_eod_price|ts_trade_day_calendar|ts_index_weight)\b",
                           lambda match: f"{match[0]} AS OF '{self.commit}'", query, flags=re.IGNORECASE)
        rows, offset = [], 0
        while True:
            page_query = f"{query} LIMIT 1000 OFFSET {offset}"
            response = self.session.get(self.URL, params={"q": page_query}, timeout=(15, 90))
            response.raise_for_status()
            payload = response.json()
            if payload.get("query_execution_status") not in ("Success", "RowLimit"):
                raise ValueError(f"DoltHub SQL error: {payload.get('query_execution_message')}")
            page = payload["rows"]
            rows.extend(page)
            if len(page) < 1000:
                return pd.DataFrame(rows, columns=[column["columnName"] for column in payload["schema"]])
            offset += len(page)


def read_bins(root, symbol, calendar):
    arrays = {field: np.fromfile(root / "features" / symbol.lower() / f"{field}.day.bin", dtype="<f4")
              for field in FIELDS}
    first = arrays["close"]
    if len(first) < 2 or not np.isfinite(first[0]) or first[0] != int(first[0]):
        raise ValueError(f"Invalid binary header: {symbol}")
    start = int(first[0])
    if start < 0 or start + len(first) - 1 > len(calendar):
        raise ValueError(f"Binary extends outside calendar: {symbol}")
    if any(len(a) != len(first) or a[0] != start for a in arrays.values()):
        raise ValueError(f"Feature lengths/offsets disagree: {symbol}")
    return pd.DataFrame({field: a[1:] for field, a in arrays.items()},
                        index=pd.DatetimeIndex(calendar[start:start + len(first) - 1]))


def normalize_increment(raw, calendar, old=None, source_anchor=None):
    """Use the existing adjusted-close scale; retain the source's amount units."""
    raw = raw.copy()
    raw["tradedate"] = pd.to_datetime(raw["tradedate"])
    if raw.duplicated("tradedate").any():
        raise ValueError("Duplicate symbol/date in source")
    raw = raw.set_index("tradedate").sort_index()
    if not raw.index.isin(calendar).all():
        raise ValueError("Source prices contain dates outside the trading calendar")
    symbol = str(raw["symbol"].iloc[0])
    if SYMBOL.fullmatch(symbol) is None:
        raise ValueError(f"Invalid symbol: {symbol!r}")
    numeric = ["open", "high", "low", "close", "volume", "adjclose", "amount"]
    raw[numeric] = raw[numeric].apply(pd.to_numeric, errors="raise")
    valid = raw["volume"].notna() & (raw["volume"] > 0)
    if (raw.loc[valid, ["open", "high", "low", "close", "adjclose"]].isna().any().any()
            or (raw.loc[valid, ["open", "high", "low", "close", "adjclose"]] <= 0).any().any()):
        raise ValueError(f"Invalid price/adjustment data: {symbol}")
    previous_close, previous_factor = np.nan, np.nan
    if old is None:
        if not valid.any():
            raise ValueError(f"No valid prices for new stock: {symbol}")
        scale = 1.0 / raw.loc[valid, "adjclose"].iloc[0]
        start = raw.index[0]
    else:
        anchor = old.loc[old["close"].notna() & old["factor"].notna()].iloc[-1]
        if source_anchor is None:
            raise ValueError(f"Missing source anchor: {symbol}")
        for field in ["open", "high", "low", "close", "volume", "adjclose", "amount"]:
            stored = anchor[field]
            if field in ("open", "high", "low", "close"):
                stored /= anchor["factor"]
            elif field == "volume":
                stored *= anchor["factor"]
            if not np.isclose(stored, float(source_anchor[field]), rtol=2e-5, atol=1e-5, equal_nan=True):
                raise ValueError(f"Historical source changed at {symbol} {anchor.name.date()} ({field}); "
                                 "review a full rebuild instead of appending")
        scale = float(anchor["close"]) / float(source_anchor["adjclose"])
        previous_close = float(source_anchor["close"])
        previous_factor = float(anchor["factor"])
        start = old.index[-1]
    after_start = calendar >= start if old is None else calendar > start
    dates = calendar[after_start & (calendar <= raw.index[-1])]
    frame = raw.reindex(dates).copy()
    invalid = frame["volume"].isna() | (frame["volume"] <= 0)
    frame.loc[invalid, numeric] = np.nan
    close = frame["close"].ffill()
    previous = close.shift(1).fillna(previous_close)
    change = close / previous - 1
    if change.between(89, 111).any():
        raise ValueError(f"Suspected 100x source price error: {symbol}")
    factor = (frame["adjclose"] / frame["close"] * scale).ffill().fillna(previous_factor)
    result = pd.DataFrame(index=dates)
    for field in ("open", "high", "low", "close"):
        result[field] = frame[field] * factor
    result["volume"] = frame["volume"] / factor
    result["factor"] = factor
    result["adjclose"] = frame["adjclose"]
    result["amount"] = frame["amount"]
    result["vwap"] = frame["amount"] / frame["volume"] * 10 * factor
    result["change"] = change.mask(invalid)
    result["symbol"] = symbol
    result.index.name = "tradedate"
    if np.isinf(result[list(FIELDS)].to_numpy()).any():
        raise ValueError(f"Infinite normalized values: {symbol}")
    return result.reset_index()


def index_intervals(weights, target):
    """Collapse consecutive identical snapshots, preserving A -> B -> A changes."""
    rows, snapshots = [], []
    for date, group in weights.groupby("trade_date", sort=True):
        members = tuple(sorted(set(group["stock_code"])))
        if not snapshots or members != snapshots[-1][1]:
            snapshots.append((pd.Timestamp(date).date(), members))
    if not snapshots:
        raise ValueError("No index membership snapshots")
    for i, (start, members) in enumerate(snapshots):
        end = snapshots[i + 1][0] - timedelta(days=1) if i + 1 < len(snapshots) else target.date()
        for code in members:
            if re.fullmatch(r"[0-9]{6}\.(SH|SZ|BJ)", code) is None:
                raise ValueError(f"Invalid constituent symbol: {code}")
            rows.append((code[-2:] + code[:6], start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    return "".join("\t".join(row) + "\n" for row in sorted(rows))


def prepare(dolt, root, work):
    old_calendar = pd.DatetimeIndex(pd.to_datetime((root / "calendars/day.txt").read_text().splitlines()))
    if old_calendar.empty or not old_calendar.is_monotonic_increasing or old_calendar.has_duplicates:
        raise ValueError("Existing calendar must be nonempty, sorted and unique")
    cutoff = old_calendar[-1].strftime("%Y-%m-%d")
    instruments = pd.read_csv(root / "instruments/all.txt", sep="\t", header=None,
                              names=["symbol", "start", "end"])
    if instruments["symbol"].duplicated().any() or not instruments["symbol"].map(lambda s: bool(SYMBOL.fullmatch(s))).all():
        raise ValueError("Invalid or duplicate symbols in all.txt")
    symbols = set(instruments["symbol"])
    target = pd.Timestamp(dolt.sql("SELECT MAX(tradedate) AS date FROM final_a_stock_eod_price").iloc[0, 0])
    if pd.isna(target) or target < old_calendar[-1]:
        raise ValueError(f"Source date {target} is older than local date {cutoff}")
    all_calendar = pd.DatetimeIndex(pd.to_datetime(dolt.sql(
        f"SELECT date FROM ts_trade_day_calendar WHERE exchange='SSE' AND is_open=1 "
        f"AND date >= '{old_calendar[0].date()}' ORDER BY date")["date"]))
    calendar = all_calendar[all_calendar <= target]
    if (not all_calendar.is_monotonic_increasing or all_calendar.has_duplicates or target not in calendar
            or not calendar[:len(old_calendar)].equals(old_calendar)):
        raise ValueError("Source and local trading calendars disagree")
    log(f"Reading incremental prices after {cutoff} (source latest: {target.date()})")
    raw = dolt.sql(f"SELECT symbol,tradedate,open,high,low,close,volume,adjclose,amount "
                   f"FROM final_a_stock_eod_price WHERE tradedate > '{cutoff}' ORDER BY symbol,tradedate")
    if raw.empty:
        log(f"No new prices; local and source latest date: {cutoff}")
        return None
    if raw.duplicated(["symbol", "tradedate"]).any():
        raise ValueError("Duplicate source symbol/date")
    if not raw["symbol"].map(lambda s: isinstance(s, str) and bool(SYMBOL.fullmatch(s))).all():
        raise ValueError("Invalid source symbol")
    for date in calendar[calendar > old_calendar[-1]]:
        count = (pd.to_datetime(raw["tradedate"]) == date).sum()
        if count < min(1000, len(symbols)):
            raise ValueError(f"Incomplete source day {date.date()}: only {count} rows")
    old_frames, anchors = {}, []
    ends = instruments.set_index("symbol")["end"].to_dict()
    for symbol in sorted(set(raw["symbol"]) & symbols):
        old = read_bins(root, symbol, old_calendar)
        if str(old.index[-1].date()) != ends[symbol]:
            raise ValueError(f"Instrument end and binary end disagree: {symbol}")
        old_frames[symbol] = old
        anchor = old[old["close"].notna() & old["factor"].notna()].index[-1]
        anchors.append(f"('{symbol}','{anchor.date()}')")
    log(f"Checking historical anchors for {len(anchors)} existing symbols")
    source_anchors = {}
    if anchors:
        # Small batches also keep the public SQL API's encoded URL below proxy limits.
        batches = []
        for offset in range(0, len(anchors), 60):
            batches.append(dolt.sql("SELECT symbol,tradedate,open,high,low,close,volume,adjclose,amount "
                                    "FROM final_a_stock_eod_price WHERE (symbol,tradedate) IN ("
                                    + ",".join(anchors[offset:offset + 60]) + ") ORDER BY symbol"))
            if offset % 600 == 0:
                log(f"Read {min(offset + 60, len(anchors))}/{len(anchors)} anchors")
        table = pd.concat(batches, ignore_index=True)
        source_anchors = table.set_index("symbol").to_dict(orient="index")
    normalized = work / "normalized"
    normalized.mkdir()
    rows = 0
    for symbol, group in raw.groupby("symbol", sort=True):
        if symbol not in symbols:
            group = dolt.sql("SELECT symbol,tradedate,open,high,low,close,volume,adjclose,amount "
                             f"FROM final_a_stock_eod_price WHERE symbol='{symbol}' "
                             f"AND tradedate >= '{old_calendar[0].date()}' ORDER BY tradedate")
        frame = normalize_increment(group, calendar, old_frames.get(symbol), source_anchors.get(symbol))
        frame.to_csv(normalized / f"{symbol}.csv", index=False)
        rows += len(frame)
    log(f"Prepared {rows} normalized rows for {raw.symbol.nunique()} symbols: {cutoff} -> {target.date()}")
    index_text = {}
    for name, code in INDEX_MAP.items():
        # MAX/GROUP BY scans millions of rows on DoltHub; use the date index's last row.
        dates = dolt.sql("SELECT trade_date FROM (SELECT trade_date FROM ts_index_weight "
                         f"FORCE INDEX (index_date) WHERE index_code='{code}' AND trade_date <= '{cutoff}' "
                         "ORDER BY trade_date DESC LIMIT 1) AS latest")
        start = pd.Timestamp(dates.iloc[0, 0]) if not dates.empty else pd.NaT
        if pd.isna(start):
            raise ValueError(f"Missing index snapshot: {name}")
        log(f"Updating {name} membership from {start.date()}")
        weights = dolt.sql("SELECT trade_date,stock_code FROM ts_index_weight FORCE INDEX (index_date) "
                           f"WHERE index_code='{code}' "
                           f"AND trade_date >= '{start.date()}' AND trade_date <= '{target.date()}' "
                           "ORDER BY trade_date,stock_code")
        history = pd.read_csv(root / "instruments" / f"{name}.txt", sep="\t", header=None,
                              names=["symbol", "start", "end"], dtype=str)
        history = history[history.start < str(start.date())].copy()
        history["end"] = history["end"].clip(upper=str(start.date() - timedelta(days=1)))
        recent = index_intervals(weights, target)
        lines = history.to_csv(sep="\t", index=False, header=False).splitlines() + recent.splitlines()
        index_text[name] = "\n".join(sorted(lines)) + "\n"
    return normalized, calendar, all_calendar, index_text


def apply_update(root, work, prepared, commit):
    normalized, calendar, all_calendar, index_text = prepared
    stage = work / "cn_data"
    log("Copying existing dataset to staging directory")
    shutil.copytree(root, stage)
    DumpDataUpdate(data_path=str(normalized), qlib_dir=str(stage), date_field_name="tradedate",
                   exclude_fields="tradedate,symbol", max_workers=4).dump()
    actual_calendar = pd.DatetimeIndex(pd.to_datetime((stage / "calendars/day.txt").read_text().splitlines()))
    if not actual_calendar.equals(calendar):
        raise ValueError("Updated calendar does not match source calendar")
    # dump_update logs worker failures instead of raising; verify every appended value.
    metadata = pd.read_csv(stage / "instruments/all.txt", sep="\t", header=None,
                           names=["symbol", "start", "end"]).set_index("symbol")
    for file in normalized.glob("*.csv"):
        expected = pd.read_csv(file, parse_dates=["tradedate"]).set_index("tradedate")
        actual = read_bins(stage, file.stem, calendar)
        if str(expected.index[-1].date()) != metadata.loc[file.stem, "end"]:
            raise ValueError(f"Failed to update instrument metadata: {file.stem}")
        np.testing.assert_allclose(actual.loc[expected.index, list(FIELDS)], expected[list(FIELDS)],
                                   rtol=2e-6, atol=1e-7, equal_nan=True, err_msg=file.stem)
        old_path = root / "features" / file.stem.lower()
        if old_path.exists():
            for field in FIELDS:
                old_bytes = (old_path / f"{field}.day.bin").read_bytes()
                with (stage / "features" / file.stem.lower() / f"{field}.day.bin").open("rb") as stream:
                    if stream.read(len(old_bytes)) != old_bytes:
                        raise ValueError(f"Historical bytes changed: {file.stem} {field}")
    for name, content in index_text.items():
        (stage / "instruments" / f"{name}.txt").write_text(content)
    (stage / "calendars/day_future.txt").write_text("".join(f"{d.date()}\n" for d in all_calendar))
    (stage / "incremental_update.json").write_text(json.dumps({
        "source": "chenditc/investment_data", "dolt_commit": commit,
        "latest_date": str(calendar[-1].date()), "updated_at": datetime.now().isoformat(),
    }, indent=2) + "\n")
    log("Staging verification passed; historical bytes unchanged")
    return stage


def publish(root, stage):
    backup = root.with_name(f"{root.name}.backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    try:
        root.rename(backup)
        stage.rename(root)
    except BaseException:
        if backup.exists() and not root.exists():
            backup.rename(root)
        raise
    pattern = re.compile(re.escape(root.name) + r"\.backup-\d{8}-\d{6}\Z")
    for previous in root.parent.iterdir():
        if previous != backup and pattern.fullmatch(previous.name) and previous.is_dir() and not previous.is_symlink():
            try:
                shutil.rmtree(previous)
                log(f"Removed older backup: {previous}")
            except OSError as error:
                log(f"Update succeeded, but could not remove older backup {previous}: {error}")
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qlib-dir", type=Path, default=Path.home() / ".qlib/qlib_data/cn_data")
    parser.add_argument("--proxy", help="Optional HTTP proxy for the public DoltHub SQL API")
    parser.add_argument("--dry-run", action="store_true", help="Prepare and verify a copy without replacing live data")
    args = parser.parse_args()
    root = args.qlib_dir.expanduser().resolve(strict=True)
    with ExitStack() as stack:
        lock = stack.enter_context((root.parent / ".cn_data-update.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log("Opening public DoltHub database at an immutable commit (no full database clone)")
        dolt = DoltHub(args.proxy)
        commit = dolt.commit
        work = Path(tempfile.mkdtemp(prefix=".cn-update-", dir=root.parent))
        log(f"Work directory: {work}")
        try:
            prepared = prepare(dolt, root, work)
            if prepared is None:
                work.rmdir()
                return
            stage = apply_update(root, work, prepared, commit)
            if args.dry_run:
                log(f"Dry run passed. Live data unchanged; verified copy: {stage}")
                return
            backup = publish(root, stage)
            log(f"Updated through {prepared[1][-1].date()}; previous dataset: {backup}")
            log(f"Normalized input retained at {prepared[0]}")
        except BaseException:
            log(f"Update stopped; diagnostic files retained at {work}")
            raise


if __name__ == "__main__":
    main()
