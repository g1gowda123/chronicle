# Changelog

All notable changes to Chronicle are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Model-free onboarding path: `chronicle demo`, `chronicle ask`, and
  `chronicle doctor` work from a core install with no GPU, model weights, video,
  or dashboard dependencies.
- Bundled SQLite demo memory generated from a deterministic pick-and-place
  scene; `chronicle ask` uses it when `--db` is omitted.
- Lazy imports for the optional perception, semantic-model, and UI layers.
- Clean-wheel and clean-sdist CI checks, including the bundled demo database.
- `chronicle doctor` opens the bundled memory and runs a real graph query, so a
  corrupt or incomplete package install is reported instead of only checking
  that the database file exists.
- Object-count questions such as `how many cups are there?` are answered from
  tracked entities, including an explicit unknown-object answer. Generic
  questions such as `how many objects are there?` count every tracked
  foreground entity.
- Generic episode questions such as `what did I do?` return the recorded
  pick-up/placement narrative instead of an abstention. If a clip ends before a
  placement is confirmed, the answer reports the interaction honestly; up to
  three recent interactions are included when several objects were handled.
- `chronicle --version` reports the installed distribution version.
- `CHANGELOG.md`, release automation, security policy, issue templates, and
  contributor documentation.

### Fixed

- `chronicle ask` and `chronicle doctor` open bundled or user-provided memories
  read-only, so they work from read-only site-packages and cannot accidentally
  mutate an existing memory during a query.

### Changed

- The Python distribution is now **`chronicle-memory`**; the import package and
  CLI remain `chronicle`. This avoids the unrelated `chronicle` package on
  PyPI.
- The public `main` branch is the stable framework entry point. Research and
  benchmark harnesses live on `research/*` and `experiment/*` branches until
  validated.

### Known limitations

- The controlled QA suites are small (two clips, 18 questions); standard
  benchmark results are still in progress.
- Official zone-token location scores are lenient. The research notes report a
  stricter support audit alongside them.
- Fine-grained action semantics such as slicing and assembling remain future
  work.
- The deterministic semantic verifier is still used when captions are disabled;
  "VLM optional" refers to caption content on grounded questions.

[Unreleased]: https://github.com/g1gowda123/chronicle/commits/main
