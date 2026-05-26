# sid_rec_runner — automated SID recording from Ultimate 64

Drives an Ultimate 64 to play `.sid` files over its REST API and tells Audacity to record each tune to its own `.wav`. Looks up each subtune's length in the HVSC `Songlengths.md5` database so it stops at the right time.

---

## One-time setup

1. **Audacity** — enable the script pipe:
   - `Edit ▸ Preferences ▸ Modules ▸ mod-script-pipe` → set to **Enabled**, restart Audacity.
   - Verify pipes exist after restart (macOS / Linux):
     ```bash
     ls /tmp/audacity_script_pipe.*
     ```
   - In `Audio Setup ▸ Recording Device`, select the audio interface you record the C64 through. The script doesn't change this.
2. **Ultimate 64** — note its IP. If you set a REST password on the device, you'll pass it with `--password`.
3. **Python deps** — already in this project's `uv` env. Nothing extra needed.

---

## Quick start

```bash
# default subtune of every .sid in a folder, 3 s tail per tune
uv run python sid_rec_runner.py \
    --host 192.168.1.50 \
    --sids sids/SID_TEST/SIDmusicTest \
    --first-only \
    --out recordings/u64
```

Recordings land in `recordings/u64/<sanitized_title>__sub<N>.wav`.

---

## What it does, per tune

1. Parses the PSID/RSID header — subtunes, default song, title.
2. MD5-hashes the file and looks the duration up in HVSC's `Songlengths.md5` (downloaded automatically on first run to `~/.cache/sid-analysis/Songlengths.md5`). Falls back to `--default-duration` if not in HVSC.
3. Tells Audacity: `New:` → `Record1stChoice:`
4. `POST http://HOST/v1/runners:sidplay?songnr=N` with the SID file as the body — the U64 plays it on its real SID chip.
5. Sleeps `length + --tail-seconds`.
6. Tells Audacity: `Stop:` → `Export2: Filename="…"` → clear tracks.
7. `PUT /v1/machine:reset` so the next tune starts from a clean state.

---

## Common invocations

```bash
# All subtunes of a single SID, cap each at 5 minutes
uv run python sid_rec_runner.py --host 192.168.1.50 \
    --sids sids/SID_TEST/SIDmusicTest/Arkanoid.sid \
    --max-duration 300

# Specific subtunes only
uv run python sid_rec_runner.py --host 192.168.1.50 \
    --sids xyz.sid --subtunes 1,3,5

# Recursively grab a folder, skip RSID tunes
uv run python sid_rec_runner.py --host 192.168.1.50 \
    --sids sids/SID_TEST --skip-rsid --first-only

# Test the U64 path only — no Audacity recording
uv run python sid_rec_runner.py --host 192.168.1.50 \
    --sids xyz.sid --first-only --no-audacity
```

---

## All options

| Flag | Default | What it does |
|---|---|---|
| `--host` | *(required)* | Ultimate 64 IP or hostname. |
| `--password` | none | `X-password` header value if you've set one on the U64. |
| `--sids` | *(required)* | One or more `.sid` files or directories (directories are walked recursively). |
| `--out` | `recordings` | Output directory for `.wav` files. |
| `--tail-seconds` | `3` | Extra seconds to keep recording after HVSC length. |
| `--between-seconds` | `1` | Pause after U64 reset before the next tune. |
| `--default-duration` | `180` | Fallback length (s) when a tune isn't in HVSC. |
| `--max-duration` | none | Cap every subtune to this many seconds. |
| `--songlengths` | `~/.cache/sid-analysis/Songlengths.md5` | Local cache path for the HVSC DB. |
| `--songlengths-url` | `https://hvsc.c64.org/.../Songlengths.md5` | Where to fetch it from on first run. |
| `--skip-rsid` | off | Skip RSID files (BASIC/Kernal-dependent — sometimes don't autoplay reliably via the REST runner). |
| `--first-only` | off | Only record the default/start subtune of each SID. |
| `--subtunes` | `all` | `all`, a single number (`3`), or a CSV list (`1,3,5`). |
| `--audacity` / `--no-audacity` | on | Toggle Audacity recording. With `--no-audacity` the script still drives the U64 — useful for smoke-testing the network path. |

---

## Output naming

`<sanitized title>__sub<N>.wav`

The title comes from the SID header. Spaces and punctuation are replaced with `_`. If the title is empty, the filename stem of the source `.sid` is used.

---

## Sharp edges

- **Audacity export uses Audacity's current export prefs.** By default that's 16-bit PCM WAV. To change to 24-bit, adjust `File ▸ Export ▸ Defaults` once before running.
- **`Record1stChoice:` records on whatever input is currently selected** in Audacity. Pick the right capture device first.
- **`machine:reset` reboots the C64.** If you have a multi-SID setup, both chips reset. That's typically what you want.
- **HVSC durations include the `+1 s` quirk** that Phosphor / ultimate64-manager apply. A tune marked `3:45` in HVSC = 226 s of recording time.

---

## Stopping early

`Ctrl-C` once. The script sends `Stop:` to Audacity and `machine:reset` to the U64, then exits.

---

## Where the bits come from

- REST endpoints: `~/Projects/ultimate64/src/lib.rs`
- HVSC songlength parser convention (and `+1 s`): `~/Projects/Phosphor/src/playlist.rs`
- PSID/RSID header layout, raw-file MD5: `~/Projects/Phosphor/src/player/sid_file.rs`
