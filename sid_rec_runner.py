#!/usr/bin/env python3
"""Auto-record SID tunes from an Ultimate 64 via REST, driving Audacity.

What it does
────────────
For every .sid file you point it at, the script:

1. Parses the PSID/RSID header (subtune count, default song, title, model hint).
2. Looks up each subtune's duration in the HVSC Songlengths.md5 database
   (downloads it on first run if missing). Falls back to ``--default-duration``
   when the tune isn't in HVSC.
3. For each requested subtune:
     a) tells Audacity to start a new project and start recording
     b) POSTs the SID to the Ultimate 64 (``runners:sidplay?songnr=N``) so the
        machine plays it natively
     c) waits ``length + --tail-seconds`` so the tail / decay is captured
     d) tells Audacity to stop + export to ``OUT/<title>__sub<N>.wav``
     e) issues a ``machine:reset`` on the U64 so the next tune starts clean

References (read directly from your project tree)
─────────────────────────────────────────────────
  ~/Projects/ultimate64/src/lib.rs       — REST endpoint shapes
                                             POST /v1/runners:sidplay?songnr=N
                                             PUT  /v1/machine:reset|pause|resume
                                             header X-password: <pw>
  ~/Projects/Phosphor/src/playlist.rs    — Songlengths.md5 parser (incl. +1 sec)
  ~/Projects/Phosphor/src/player/sid_file.rs — PSID/RSID header layout, md5 of raw

Audacity automation
───────────────────
Uses the ``mod-script-pipe`` plugin (enabled by default in modern Audacity).
On macOS/Linux it exposes two FIFOs under /tmp; on Windows it uses named pipes.
If the pipes aren't there, enable ``Edit ▸ Preferences ▸ Modules ▸ mod-script-pipe = enabled``
and restart Audacity once.

Usage
─────
  uv run python sid_rec_runner.py \\
      --host 192.168.1.50 --password secret \\
      --sids sids/SID_TEST/SIDmusicTest/*.sid \\
      --out recordings/ \\
      --tail-seconds 3 \\
      --default-duration 180
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import struct
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ──────────────────────────────────────────────────────────────────────────────
#  Ultimate 64 REST client
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_SONGLENGTHS_URL = (
    "https://hvsc.c64.org/download/C64Music/DOCUMENTS/Songlengths.md5"
)


class Ultimate64:
    """Tiny REST client matching what ~/Projects/ultimate64/src/lib.rs exposes."""

    def __init__(self, host: str, password: str | None = None, timeout: float = 10.0):
        self.base = f"http://{host}/v1"
        self.headers = {"X-password": password} if password else {}
        self.timeout = timeout

    def _request(self, method: str, path: str, body: bytes | None = None) -> bytes:
        req = urllib.request.Request(
            f"{self.base}/{path}", data=body, method=method, headers=self.headers
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.read()

    def info(self) -> str:
        return self._request("GET", "info").decode("utf-8", "replace")

    def reset(self) -> None:
        self._request("PUT", "machine:reset")

    def pause(self) -> None:
        self._request("PUT", "machine:pause")

    def resume(self) -> None:
        self._request("PUT", "machine:resume")

    def sid_play(self, sid_bytes: bytes, songnr: int | None = None) -> None:
        path = "runners:sidplay"
        if songnr is not None:
            path += f"?songnr={songnr}"
        self._request("POST", path, sid_bytes)


# ──────────────────────────────────────────────────────────────────────────────
#  PSID / RSID header parser  (subset matching Phosphor's sid_file.rs)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SidInfo:
    path: Path
    raw: bytes
    md5: str
    magic: str                # "PSID" or "RSID"
    is_rsid: bool
    version: int
    songs: int                # total subtunes
    start_song: int           # 1-based default subtune
    title: str
    author: str
    released: str

    def safe_stem(self) -> str:
        s = self.title.strip() or self.path.stem
        return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or self.path.stem


def parse_sid(path: Path) -> SidInfo:
    data = path.read_bytes()
    if len(data) < 0x76:
        raise ValueError(f"{path}: too short to be a SID file")
    magic = data[0:4].decode("ascii", "replace")
    if magic not in ("PSID", "RSID"):
        raise ValueError(f"{path}: not a PSID/RSID file (magic={magic!r})")

    (version,)      = struct.unpack(">H", data[4:6])
    (songs,)        = struct.unpack(">H", data[14:16])
    (start_song,)   = struct.unpack(">H", data[16:18])

    def cstr(off: int, n: int) -> str:
        chunk = data[off:off + n]
        nul = chunk.find(b"\x00")
        if nul >= 0:
            chunk = chunk[:nul]
        return chunk.decode("ascii", "replace").strip()

    return SidInfo(
        path=path,
        raw=data,
        md5=hashlib.md5(data).hexdigest(),
        magic=magic,
        is_rsid=(magic == "RSID"),
        version=version,
        songs=max(1, songs),
        start_song=max(1, start_song),
        title=cstr(0x16, 32),
        author=cstr(0x36, 32),
        released=cstr(0x56, 32),
    )


# ──────────────────────────────────────────────────────────────────────────────
#  HVSC Songlengths.md5
# ──────────────────────────────────────────────────────────────────────────────


def _parse_songlength_time(token: str) -> int | None:
    """'mm:ss', 'mm:ss.xxx', or 'mm:ss(G)' -> whole seconds. Mirrors Phosphor."""
    token = token.strip().split("(", 1)[0]
    if ":" not in token:
        return None
    m, s = token.split(":", 1)
    s = s.split(".", 1)[0]
    try:
        return int(m) * 60 + int(s)
    except ValueError:
        return None


def load_songlengths(path: Path) -> dict[str, list[int]]:
    """Parse HVSC Songlengths.md5 → {md5: [duration_seconds, …]}.

    Mirrors Phosphor's loader (playlist.rs) including the ``+1 second`` quirk.
    """
    out: dict[str, list[int]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith((";", "#", "[")):
                continue
            if "=" not in line:
                continue
            md5_str, _, times_str = line.partition("=")
            md5_str = md5_str.strip()
            if len(md5_str) != 32:
                continue
            durations = []
            for tok in times_str.split():
                d = _parse_songlength_time(tok)
                if d is not None:
                    durations.append(d + 1)  # Phosphor convention
            if durations:
                out[md5_str.lower()] = durations
    return out


def ensure_songlengths(local: Path, url: str) -> Path:
    if local.exists() and local.stat().st_size > 0:
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"[hvsc] downloading songlengths → {local}  (from {url})")
    with urllib.request.urlopen(url, timeout=60) as r:
        local.write_bytes(r.read())
    print(f"[hvsc] saved {local.stat().st_size} bytes")
    return local


# ──────────────────────────────────────────────────────────────────────────────
#  Audacity mod-script-pipe driver
# ──────────────────────────────────────────────────────────────────────────────


class Audacity:
    """Talk to Audacity over its mod-script-pipe FIFOs.

    Pipes (macOS / Linux):  /tmp/audacity_script_pipe.to.<uid> (write)
                            /tmp/audacity_script_pipe.from.<uid> (read)
    Windows:                \\\\.\\pipe\\ToSrvPipe / FromSrvPipe
    """

    def __init__(self):
        if sys.platform == "win32":
            self.to_path = r"\\.\pipe\ToSrvPipe"
            self.from_path = r"\\.\pipe\FromSrvPipe"
        else:
            self.to_path = f"/tmp/audacity_script_pipe.to.{os.getuid()}"
            self.from_path = f"/tmp/audacity_script_pipe.from.{os.getuid()}"
        self._to = None
        self._from = None

    def connect(self) -> None:
        if not os.path.exists(self.to_path):
            raise RuntimeError(
                f"Audacity pipe not found at {self.to_path}. Enable "
                "Edit ▸ Preferences ▸ Modules ▸ mod-script-pipe and restart."
            )
        self._to = open(self.to_path, "w")
        self._from = open(self.from_path, "r")

    def close(self) -> None:
        if self._to: self._to.close()
        if self._from: self._from.close()
        self._to = self._from = None

    def cmd(self, command: str) -> str:
        """Send a script command, read the reply (terminated by blank line)."""
        if self._to is None:
            self.connect()
        assert self._to and self._from
        self._to.write(command + "\n")
        self._to.flush()
        reply_lines = []
        while True:
            line = self._from.readline()
            if line in ("\n", ""):
                if reply_lines:
                    break
                continue
            reply_lines.append(line.rstrip("\n"))
        return "\n".join(reply_lines)

    # High-level helpers ----------------------------------------------------
    def new_project(self) -> None:
        self.cmd("New:")

    def record(self) -> None:
        # Record1stChoice: starts on the first available track, or a new one.
        self.cmd("Record1stChoice:")

    def stop(self) -> None:
        self.cmd("Stop:")

    def export(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Export2 takes a quoted filename.
        self.cmd(f'Export2: Filename="{out_path.resolve()}" NumChannels=2')

    def select_all(self) -> None:
        self.cmd("SelectAll:")

    def remove_tracks(self) -> None:
        self.cmd("RemoveTracks:")


# ──────────────────────────────────────────────────────────────────────────────
#  Driver
# ──────────────────────────────────────────────────────────────────────────────


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for p in patterns:
        path = Path(p)
        if path.is_dir():
            out.extend(sorted(path.rglob("*.sid")))
        else:
            out.append(path)
    return [p for p in out if p.suffix.lower() == ".sid"]


def subtune_durations(info: SidInfo, db: dict[str, list[int]],
                      default_duration: int, max_duration: int | None) -> list[int]:
    """Return duration (s) for every subtune, falling back to default."""
    listed = db.get(info.md5.lower())
    out: list[int] = []
    for i in range(info.songs):
        d = listed[i] if listed and i < len(listed) else default_duration
        if max_duration is not None:
            d = min(d, max_duration)
        out.append(d)
    return out


def run(args: argparse.Namespace) -> None:
    sid_paths = expand_inputs(args.sids)
    if not sid_paths:
        print("No .sid files found.", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sl_path = ensure_songlengths(Path(args.songlengths), args.songlengths_url)
    db = load_songlengths(sl_path)
    print(f"[hvsc] {len(db)} entries in songlength DB")

    u64 = Ultimate64(args.host, args.password)
    try:
        print(f"[u64] info: {u64.info().strip()}")
    except Exception as e:
        print(f"[u64] WARNING: info() failed ({e}); continuing anyway")

    audacity: Audacity | None = None
    if args.audacity:
        audacity = Audacity()
        try:
            audacity.connect()
            print("[audacity] connected to mod-script-pipe")
        except Exception as e:
            print(f"[audacity] could not connect: {e}", file=sys.stderr)
            sys.exit(3)

    selected_only = (
        None if args.subtunes == "all" else
        ([int(x) for x in args.subtunes.split(",")] if "," in args.subtunes
         else [int(args.subtunes)] if args.subtunes.isdigit()
         else None)
    )

    try:
        for sid_path in sid_paths:
            try:
                info = parse_sid(sid_path)
            except Exception as e:
                print(f"[skip] {sid_path}: {e}", file=sys.stderr)
                continue

            if info.is_rsid and args.skip_rsid:
                print(f"[skip] {sid_path.name}: RSID and --skip-rsid set")
                continue

            durations = subtune_durations(info, db, args.default_duration, args.max_duration)
            md5_status = "HVSC" if info.md5.lower() in db else "fallback"

            print()
            print(f"=== {info.title or info.path.stem} ===")
            print(f"     path:  {info.path}")
            print(f"     md5:   {info.md5}  ({md5_status})")
            print(f"     songs: {info.songs}  start={info.start_song}  "
                  f"author={info.author!r}")

            songs_to_play = (
                selected_only if selected_only is not None
                else ([info.start_song] if args.first_only
                      else list(range(1, info.songs + 1)))
            )

            for song_no in songs_to_play:
                if song_no < 1 or song_no > info.songs:
                    print(f"  [skip subtune {song_no}] out of range")
                    continue
                length = durations[song_no - 1]
                total = length + args.tail_seconds
                out_path = out_dir / f"{info.safe_stem()}__sub{song_no}.wav"

                print(f"  ▶ subtune {song_no}/{info.songs}  length={length}s  "
                      f"(total {total}s)  → {out_path.name}")

                if audacity:
                    audacity.new_project()
                    time.sleep(0.4)
                    audacity.record()

                try:
                    u64.sid_play(info.raw, song_no)
                except Exception as e:
                    print(f"     [u64] sid_play failed: {e}", file=sys.stderr)
                    if audacity:
                        audacity.stop()
                    continue

                try:
                    time.sleep(total)
                except KeyboardInterrupt:
                    print("\n  ⏹ interrupted — stopping cleanly")
                    if audacity:
                        audacity.stop()
                    u64.reset()
                    raise

                if audacity:
                    audacity.stop()
                    time.sleep(0.3)
                    audacity.export(out_path)
                    time.sleep(0.3)
                    audacity.select_all()
                    audacity.remove_tracks()

                # Reset C64 to clear SID state before the next tune
                try:
                    u64.reset()
                except Exception as e:
                    print(f"     [u64] reset failed: {e}", file=sys.stderr)

                if args.between_seconds > 0:
                    time.sleep(args.between_seconds)
    finally:
        if audacity:
            audacity.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host", required=True, help="Ultimate 64 IP/hostname")
    p.add_argument("--password", default=None, help="X-password header value if set on U64")
    p.add_argument("--sids", nargs="+", required=True,
                   help="One or more .sid files or directories (recursive)")
    p.add_argument("--out", default="recordings",
                   help="Output directory for .wav files (default: ./recordings)")
    p.add_argument("--tail-seconds", type=float, default=3.0,
                   help="Extra seconds to keep recording after HVSC length (default: 3)")
    p.add_argument("--between-seconds", type=float, default=1.0,
                   help="Quiet pause between tunes after reset (default: 1)")
    p.add_argument("--default-duration", type=int, default=180,
                   help="Fallback length when tune not in HVSC (default: 180)")
    p.add_argument("--max-duration", type=int, default=None,
                   help="Cap any subtune length to this many seconds")
    p.add_argument("--songlengths",
                   default=str(Path.home() / ".cache/sid-analysis/Songlengths.md5"),
                   help="Local path to HVSC Songlengths.md5 (downloaded if absent)")
    p.add_argument("--songlengths-url", default=DEFAULT_SONGLENGTHS_URL,
                   help="URL to fetch Songlengths.md5 from if not cached")
    p.add_argument("--skip-rsid", action="store_true",
                   help="Skip RSID files (BASIC / KERNAL-dependent tunes)")
    p.add_argument("--first-only", action="store_true",
                   help="Only play the default/first subtune of each SID")
    p.add_argument("--subtunes", default="all",
                   help='Which subtunes to play: "all", a number "1", or a list "1,3,5"')
    p.add_argument("--audacity", action=argparse.BooleanOptionalAction, default=True,
                   help="Drive Audacity for recording (default on; --no-audacity to disable)")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
