# Releasing Chronicle

Chronicle is distributed as the Python package **`chronicle-memory`**. The import
package and CLI are still named `chronicle`.

## One-time PyPI setup

The release workflow uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/),
so no API token is stored in GitHub.

1. Create or claim the `chronicle-memory` project on PyPI.
2. Add a trusted publisher with:
   - owner: `g1gowda123`
   - repository: `chronicle`
   - workflow: `release.yml`
   - environment: `pypi`
3. In the GitHub repository, create an environment named `pypi`. Requiring a
   reviewer is recommended.

Until that setup exists, install the current release from `main`:

```bash
pip install "git+https://github.com/g1gowda123/chronicle.git@main"
```

## Release checklist

1. Start from a green `main` and a clean working tree.
2. Update the version in `pyproject.toml` and `CITATION.cff`.
3. Move the relevant entries in `CHANGELOG.md` from `Unreleased` to the new
   version and add the release date.
4. Run the local checks:

   ```bash
   python -m pytest
   ruff check .
   python -m build
   python -m twine check dist/*
   ```

5. Install the built wheel in a clean virtual environment and run the model-free
   smoke test:

   ```bash
   python -m venv /tmp/chronicle-release
   /tmp/chronicle-release/bin/pip install dist/*.whl
   /tmp/chronicle-release/bin/chronicle demo
   /tmp/chronicle-release/bin/chronicle ask "where is my cup?" --llm template
   ```

6. Commit the version bump, then tag and push:

   ```bash
   git tag -a v0.1.0 -m "Chronicle 0.1.0"
   git push origin v0.1.0
   ```

7. The `Release` workflow builds the sdist/wheel, publishes to PyPI through
   Trusted Publishing, and creates a GitHub release. Check the workflow and the
   PyPI project page.
8. Verify the public install:

   ```bash
   python -m venv /tmp/chronicle-pypi
   /tmp/chronicle-pypi/bin/pip install chronicle-memory
   /tmp/chronicle-pypi/bin/chronicle doctor
   ```

## Versioning

- Patch releases: bug fixes and documentation that do not change public
  behaviour.
- Minor releases: new framework capabilities, new CLI commands, or compatible
  API additions.
- Major releases: breaking API or data-format changes.

The `main` branch is the stable framework entry point. Experimental benchmark
harnesses live on `experiment/*` branches and are not part of the package until
they are validated.
