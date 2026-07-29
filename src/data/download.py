"""
download.py
===========
Fetch publicly available LIGO strain data from the Gravitational-Wave Open
Science Center (GWOSC) for observed runs O1, O2, and O3.

Usage
-----
python -m src.data.download --runs O1 O2 O3 --out data/raw
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import requests

log = logging.getLogger(__name__)


# ── GWOSC catalogue ──────────────────────────────────────────────────────────

GWOSC_API = "https://gwosc.org/eventapi/json/query/"
GWOSC_STRAIN = "https://gwosc.org/eventapi/json/GWTC/"

CONFIRMED_EVENTS: dict[str, list[str]] = {
    "O1": ["GW150914", "GW151012", "GW151226"],
    "O2": [
        "GW170104", "GW170608", "GW170729",
        "GW170809", "GW170814", "GW170817",
        "GW170818", "GW170823",
    ],
    "O3": [
        "GW190408", "GW190412", "GW190425",
        "GW190521", "GW190814", "GW191105",
        "GW200105", "GW200115",
    ],
}

DETECTORS = ["H1", "L1"]   # Hanford, Livingston


# ── helpers ──────────────────────────────────────────────────────────────────

def _event_url(event_name: str, detector: str = "H1") -> str | None:
    """Return the HDF5 URL for a confirmed event via GWOSC JSON API."""
    url = f"https://gwosc.org/eventapi/json/event/{event_name}/"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        for version in data.get("events", {}).values():
            for fmt in version.get("strain", []):
                if (
                    fmt.get("detector") == detector
                    and fmt.get("format") == "hdf5"
                    and "4096" in fmt.get("url", "")   # prefer 4096 Hz
                ):
                    return fmt["url"]
    except Exception as exc:
        log.warning("Could not fetch URL for %s/%s: %s", event_name, detector, exc)
    return None


def _download_file(url: str, dest: Path, retries: int = 3) -> bool:
    """Stream-download *url* to *dest*, skipping if already present."""
    if dest.exists():
        log.info("  Already downloaded: %s", dest.name)
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = 100 * downloaded / total
                            print(f"\r  {dest.name}: {pct:.1f}%", end="", flush=True)
            print()
            log.info("  Saved: %s  (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
            return True
        except Exception as exc:
            log.warning("  Attempt %d failed: %s", attempt + 1, exc)
            time.sleep(2 ** attempt)
    return False


# ── noise segments ────────────────────────────────────────────────────────────

NOISE_SEGMENTS: dict[str, list[tuple[int, int]]] = {
    # GPS start/end of quiet noise periods (not containing any confirmed event)
    "O1": [(1126259200, 1126263296), (1127271617, 1127275713)],
    "O2": [(1164556817, 1164560913), (1167545600, 1167549696)],
    "O3": [(1238166018, 1238170114), (1249852257, 1249856353)],
}


def _fetch_noise_hdf5(
    gps_start: int,
    gps_end: int,
    detector: str,
    out_dir: Path,
) -> None:
    """Download a noise segment from GWOSC bulk data API."""
    filename = f"{detector}_{gps_start}_{gps_end}_noise.hdf5"
    dest = out_dir / "noise" / filename
    if dest.exists():
        log.info("  Noise file exists: %s", filename)
        return

    url = (
        "https://gwosc.org/archive/data/"
        f"{detector}_O3a_16KHZ_R1/{gps_start}-{gps_end}.hdf5"
    )
    # Use gwpy TimeSeries.fetch_open_data as fallback
    try:
        from gwpy.timeseries import TimeSeries
        ts = TimeSeries.fetch_open_data(
            detector, gps_start, gps_end, sample_rate=4096, verbose=False
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        ts.write(str(dest), format="hdf5", overwrite=True)
        log.info("  Fetched noise segment via gwpy: %s", filename)
    except Exception as exc:
        log.warning("  gwpy fetch failed (%s); trying direct URL…", exc)
        _download_file(url, dest)


# ── main ─────────────────────────────────────────────────────────────────────

def download_events(
    runs: List[str],
    out_dir: Path,
    detectors: List[str] | None = None,
) -> None:
    """Download HDF5 strain files for confirmed events in *runs*."""
    detectors = detectors or DETECTORS
    for run in runs:
        events = CONFIRMED_EVENTS.get(run, [])
        log.info("Run %s — %d events", run, len(events))
        for event in events:
            for det in detectors:
                log.info("  %s / %s", event, det)
                url = _event_url(event, det)
                if url:
                    dest = out_dir / run / "events" / f"{event}_{det}.hdf5"
                    _download_file(url, dest)
                else:
                    log.warning("  No URL found for %s / %s", event, det)


def download_noise(runs: List[str], out_dir: Path) -> None:
    """Fetch quiet background noise segments for each run."""
    for run in runs:
        for det in DETECTORS:
            for gps_start, gps_end in NOISE_SEGMENTS.get(run, []):
                log.info("Noise %s %s %d–%d", run, det, gps_start, gps_end)
                _fetch_noise_hdf5(gps_start, gps_end, det, out_dir / run)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download LIGO open data from GWOSC")
    p.add_argument("--runs", nargs="+", default=["O1", "O2", "O3"],
                   help="Observing runs to download (default: O1 O2 O3)")
    p.add_argument("--out", default="data/raw", help="Output directory")
    p.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    p.add_argument("--no-noise", action="store_true", help="Skip noise segments")
    p.add_argument("--no-events", action="store_true", help="Skip event strain files")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.no_events:
        log.info("=== Downloading event strain files ===")
        download_events(args.runs, out, args.detectors)

    if not args.no_noise:
        log.info("=== Downloading noise segments ===")
        download_noise(args.runs, out)

    log.info("Download complete.  Data saved to: %s", out.resolve())


if __name__ == "__main__":
    main()
