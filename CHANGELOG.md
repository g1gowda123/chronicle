# Changelog

All notable changes to Chronicle are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Model-free onboarding path: `chronicle demo`, `chronicle ask`, and
  `chronicle doctor` work from a core install with no GPU, model weights, video,
  or dashboard dependencies.
- Bundled SQLite demo memory generated from a 10.6-second deterministic
  pick-and-place scene with four objects; `chronicle ask` uses it when `--db`
  is omitted.
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
- `when did I last see X?` and `when did I first see X?` are answered from the
  graph with a timestamp; the inventory answer now uses singular grammar for a
  one-entity memory.
- `chronicle eval` runs a 45-case deterministic product suite against the
  bundled memory and prints a score; it covers location, support surfaces,
  events, first/last-seen times, temporal-before questions, counts, appearance,
  surface inventories and abstention behavior without models or a judge.
- `chronicle --version` reports the installed distribution version, and
  `chronicle doctor` includes it in both text and JSON reports.
- `CHANGELOG.md`, release automation, security policy, issue templates, and
  contributor documentation.

### Fixed

- Event timelines are filtered to edges incident on the queried entity, so
  objects that share a support surface no longer inherit each other's
  pick-up/placement history.
- Plural detector labels such as `keys` now use correct agreement and pronouns
  in location, appearance and event answers.
- `where was X before it was picked up?` resolves the pick-up event instead of
  falling back to the latest placement.
- Event questions about an absent named object abstain instead of summarizing
  unrelated objects, and `what is on the <surface>?` answers from terminal
  support anchors.
- Surface-inventory answers report the most recent observation of each tracked
  object instead of the object's original placement time.
- Episode summaries such as `what did I see?` use singular nouns for one item
  and also accept `which objects/things did I see?`.
- Short temporal paraphrases such as `where was the cup before it was placed?`
  resolve the placement event instead of abstaining.
- `what happened first?` and `what happened last?` select the earliest or most
  recent interaction instead of returning the generic newest-first overview.
- Location questions without a recorded spatial fact now abstain explicitly
  instead of presenting an unrelated action or state as a location.
- Hyphenated detector labels such as `joy-con` now canonicalize with
  `controller`, so count questions group detector aliases correctly.
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
- Template-only queries no longer construct the semantic engine. On the bundled
  demo memory, `chronicle ask "where is my cup?"` dropped from ~4.4 s to ~0.5 s
  while keeping the same graph answer.

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
