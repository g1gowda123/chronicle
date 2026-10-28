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
- `CHANGELOG.md`, release automation, security policy, issue templates, and
  contributor documentation.

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
