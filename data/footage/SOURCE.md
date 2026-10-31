# Real clip provenance

## Git LFS clips

`aria_cook_45s.mp4`, `aria_play_45s.mp4`, `aria_walk_60s.mp4`, and
`holoassist_switch_34s.mp4` are tracked through Git LFS. A plain checkout
without `git-lfs` leaves 132-byte pointer files. Restore the real media with:

```bash
python scripts/fetch_lfs_clips.py --dry-run
python scripts/fetch_lfs_clips.py --all
```

The fetcher reads the pointer's SHA-256 and size, downloads the media from
GitHub's `media.githubusercontent.com` endpoint, and verifies both before
replacing the pointer. Downloaded media remains ignored by git and is not
committed to the repository.

## HoloAssist

- Dataset: **HoloAssist** — <https://holoassist.github.io/>
- Session: `z210-sep-29-22-switch`
- Device: Microsoft HoloLens 2 (first-person egocentric video)
- Source asset: `video_compress.tar` on the HoloAssist Azure release bucket
- Source member: `z210-sep-29-22-switch/Export_py/Video_compress.mp4`
- Original resolution / rate: 454×256, 29.943 fps
- Local raw download: `holoassist_raw.mp4` (not trimmed)
- Demo clip: `holoassist_switch_34s.mp4`, first 34 s starting at 6 s of the session
- Annotations: `holoassist_switch_34s.json` (action windows, zone, calibration ground truth)
- License: **CDLA-Permissive-2.0**

The clip shows a person assembling Joy-Con controllers onto a Nintendo Switch /
comfort grip on a table. It is real smart-glasses footage, but it is a tabletop
assembly task rather than a room walkthrough.

The HoloAssist source archive is retrieved with HTTP range requests; only the
needed tar member is read. The repository keeps only the trimmed clip and its
small metadata file. (The old `python -m chronicle fetch-real` note was stale;
that command is not implemented in this worktree.)

## Additional Aria Gen 2 clips

The kitchen (`cook_0`) and living-room (`play_2`) clips are genuine Meta Aria
Gen 2 Pilot footage under **CC BY-NC 4.0**. Their provenance, source windows,
license links and fetch commands are in:

- [`cook/SOURCE.md`](cook/SOURCE.md)
- [`play/SOURCE.md`](play/SOURCE.md)
- [`walking/SOURCE.md`](walking/SOURCE.md)

Like the walking clip, they are locally trimmed research windows and are not
redistributed. The full source datasets remain with Meta.
