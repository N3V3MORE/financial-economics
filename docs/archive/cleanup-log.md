# Cleanup log

A record of the October 2026 tidy-up, one entry per change, in the order the changes were made.

## Ground rules

- The eight forecasting files whose hashes guard the saved reconstruction caches stay byte-for-byte unchanged: `paper_sources.py`, `paper.py`, `forecasting.py`, `paper_models.py`, `paper_networks.py`, `paper_networks_log.py`, `paper_gru.py` and `gru_forecasting.py`.
- `paper_revised` and `proxy_study` hash every module in the package into their frozen protocols, so any source edit means a protocol frozen before this cleanup no longer matches. None existed in the `outputs/` folder of the machine used for this cleanup. Freeze a new protocol after this cleanup rather than reusing an older one.
- Each change is checked with `uv run --frozen ruff check` and `uv run --frozen pytest -q` before the next one starts.

Baseline before any change (branch `volatility-project-checkpoint`, commit `e3a3f05`): ruff reports no problems and 392 tests pass in 79 s.

## Changes

### 1. Removed stale E402 ignores from the ruff config

File: `pyproject.toml`.

The per-file ignores for `*_report.py` and `paper_figures.py` were there because those modules import pyplot after calling `matplotlib.use("Agg")`. Ruff already treats imports that follow `matplotlib.use(...)` as allowed, so the ignores matched nothing. Removed the comment and both entries. `ruff check` still passes.

### 2. Declaring scipy directly: not done

`paper_models.py`, `paper_realistic.py`, `volatility_proxy.py` and `scripts/verify_protection.py` import scipy, but `pyproject.toml` does not list it. It only gets installed because statsmodels depends on it. Adding it means re-locking, and the uv on this machine (0.13.3) is newer than the one CI pins (0.11.15). Re-locking with it bumped the lockfile format from revision 3 to 5 and rewrote markers on cffi, pytest, rocm and torch. No versions changed, but CI's older uv may not read the new format. `pyproject.toml` and `uv.lock` were restored. Next time the lock is regenerated with uv 0.11.15, add `"scipy>=1.14,<2"` to the dependencies.

Restoring `pyproject.toml` also undid change 1, because that edit had not been committed yet. Change 1 was re-applied by hand and checked again.

### 3. Matched the contributor commands to CI

File: `CONTRIBUTING.md`, steps 3 and 4.

The guide said `uv sync --frozen` and `uv run pytest -q`. CI installs the `reconstruction` extra, runs ruff, and runs pytest with `-W error`. Without the extra, `arch` is missing and all of `tests/test_paper_models.py` is skipped without failing. Steps 3 and 4 now give the same commands CI runs. A reviewer noticed the first version of step 4 left `--extra reconstruction` off the pytest command, and it was added back.

### 4. Used the shared timestamp and hash helpers in two downloaders

Files: `src/financial_economics/paper_macro_sources.py`, `src/financial_economics/wrds_source.py`.

Both modules wrote out `datetime.now(UTC).isoformat()` themselves, and `paper_macro_sources` also wrote out `sha256(path.read_bytes()).hexdigest()`. `provenance.py` already has `utc_now()` and `file_digest()`, which return exactly those values, and the other modules use them. Both files now call those helpers, and the imports they no longer need are gone. The manifests they write come out the same as before. Ruff passes, and so do the 49 tests in `test_paper_macro_sources.py`, `test_wrds_source.py` and `test_paper_source_alternatives.py`.

### 5. One helper for "git commit, or unavailable"

Files: `src/financial_economics/provenance.py`, `protection_report.py`, `prediction_report.py`.

The protection and prediction reports had the same five-line `try`/`except` that records the git commit and dirty flag, falling back to `"unavailable"` and `None` when git can't be read. That block is now `provenance.git_state()`, and both reports call it. Both modules no longer import `subprocess`. The other modules that call `git_head()` and `git_dirty()` directly have no fallback on purpose, so they were left alone. Ruff passes, and so do the 17 protection and prediction tests.

### 6. Shared helpers for code and output-file digests

Files: `src/financial_economics/provenance.py`, `paper_revised.py`, `proxy_study.py`, `protection_report.py`, `prediction_report.py`, `paper_extension.py`, `tests/test_paper_revised.py`, `tests/test_proxy_study.py`.

Two dictionary expressions had been copied into four modules each, though not the same four:

- the SHA-256 of every `.py` file in the package (the code hashes that frozen protocols are checked against), in `paper_revised`, `protection_report`, `prediction_report` and `paper_extension`;
- the SHA-256 of every file under a finished run folder, keyed by its relative path with forward slashes, in `paper_revised`, `proxy_study`, `prediction_report` and `paper_extension`.

They are now `provenance.package_code_digests()` and `provenance.directory_digests(output)`. `paper_revised._code_identity` is gone, and `proxy_study` gets the helper from `provenance` instead of importing it from `paper_revised`. `provenance.py` sits in the same folder as the old callers, so the digests come out the same. Three tests in two files monkeypatched `_code_identity` by name. They now patch `package_code_digests`, and nothing else in them changed.

Left alone on purpose: `paper_reconstruction.py` builds its file map with `str(...)` rather than `.as_posix()`. On Windows that writes backslashes, and saved runs are checked against those keys. The flat `iterdir()` maps in other modules were also left as they are.

Full suite after this change: 399 passed, 4 failed. All four failures are in `test_paper_reconstruction.py`. They come from uncommitted MIOpen GRU edits to `gru_forecasting.py` on branch `miopen-gru-training`, which changed that file's guarded hash. None of them involve a file this cleanup touched.

### 7. Shared helper for checking a run folder against its recorded hashes

Files: `src/financial_economics/provenance.py`, `paper_revised.py`, `volatility_proxy.py`.

`paper_revised._read_stage` and `volatility_proxy.load_accepted_proxy` had the same loop over a run's recorded files. It refuses any file that resolves outside the run folder or whose SHA-256 no longer matches. The two copies differed only in their error message. The loop is now `provenance.verify_files(directory, files, message)`, and each caller passes its original message unchanged. The single-file checks elsewhere in `volatility_proxy.py` were not touched. Ruff passes, and so do the 31 tests in `test_volatility_proxy.py`, `test_paper_revised.py` and `test_proxy_study.py`.

### Review of changes 1 to 7

A read-only review compared each new helper with the code it replaced. It checked ordering, dictionary keys, path separators, exception types and messages, and when the run-folder digests are taken relative to writing `run.json`. It found no differences. All eight edited modules import cleanly, and the 108 tests in the ten affected test files pass. A second check of this log against the diff fixed three overstatements: the test count in entry 6, the "same commands" claim in entry 3, and which tests skip without `arch`.

### 8. README file map brought up to date

File: `README.md`, the "Find your way around" section only.

Five modules were missing from the file map: `wrds_credentials.py`, `paper_macro_sources.py`, `paper_revised.py`, `volatility_proxy.py` and `proxy_study.py`. Each now has a one-line entry in its group, worded from the module's own docstring. The "Other files" table also called `supplement/Tbill-V4.html` the "rendered page" of `Tbill V4.Rmd`. It isn't one: its footer says it is a static companion updated from the CSV and that the R Markdown was never run. The row now says "a static HTML summary". Other README changes in the working tree come from the MIOpen GRU work and were not touched.

### 9. Moved the run configs into `configs/`

Files moved: the ten run-config TOMLs (`study`, `study-commodities`, `paper`, `paper-extension`, `paper-reconstruction`, `paper-realistic`, `paper-realistic-ief`, `prediction`, `protection`, `protection-long`) went from the repository root to `configs/`. `pyproject.toml` stays at the root.

References updated:
- argparse defaults in `cli.py`, `paper_realistic.py`, `paper_reconstruction.py` (that one line only), `paper_revised.py` and `proxy_study.py`;
- the real-file paths in `tests/test_forecasting.py`, `tests/test_paper_evaluation.py` and `tests/test_protection.py`;
- about three dozen commands and mentions in `README.md`, plus its file map and "Other files" table;
- `CONTRIBUTING.md`, `docs/methodology.md` and `docs/results-explained.md`.

The root folder went from 22 visible entries to 13. Run records store a hash of each config's content, never its path, so saved runs still verify. Each moved file's content is byte-identical to the committed version. The files were moved with a plain `mv`, not `git mv`, so nothing was staged in a checkout someone else was committing from. Git will show the move as ten deletions plus a new `configs/` folder until it is staged.

Left as is: `paper.py` (hash-frozen) still says "such as paper.toml" in one error message, which is still the right file name. `handover.md` lines 84-85 were updated last, once the MIOpen GRU work finished with that file.

After the move: ruff passes, `fecon --help` starts, and the suite gives 399 passed with the same 4 `gru_forecasting.py` failures as before. Those four were fixed by the MIOpen GRU work itself, which was committed separately as `4001531`; see the final check below.

## Final check

Run on branch `miopen-gru-training` at `4001531` (the MIOpen GRU commit) with all nine cleanup changes still uncommitted on top:

- `uv run --frozen ruff check` passes.
- `uv run --frozen --extra reconstruction pytest -q -W error`, the command CI runs, gives 406 passed, 0 failed.
- A read-only search of every tracked file plus `configs/` found no reference to an old root config path, apart from the `paper.py` message noted above and two comments inside `configs/paper-realistic-ief.toml` that name a sibling file. All five `--help` entry points start, and every argparse default points at a file that exists.
- A second check of this log against the diff for changes 8 and 9 corrected the root-folder count and the number of README edits.

Nothing from this cleanup is committed. Until it is staged, git shows the config move as ten deleted root files plus an untracked `configs/` folder.

## Looked at and left alone

- Dead code: a full search found none. Functions used only by tests or by their own module's entry point are hooks the tests rely on, or documented manual entry points such as `paper_fx.download_fx`.
- Results text repeated across `README.md`, `handover.md` and `docs/final-study.md`, such as the matched-study table and the proxy-pilot numbers. `docs/final-study.md` is the fullest copy. Whether to shorten the others is up to the authors, so they were not changed.
- One-line repeats such as the `run.json` write, the matplotlib `Agg` setup and the TOML load. A helper would save little and touch many files.
- The bootstrap and Holm blocks in `protection_inference.py` and `prediction_inference.py` look alike but build their masks differently. They feed reported statistics, so they were not merged.
- The `_prepared` test builder shared by `test_paper_networks.py` and `test_paper_networks_log.py`. Two near-copies don't justify a shared fixture.
- The ruff ignore list for the eight frozen files is wider than it currently needs to be. It guards against a future ruff release flagging those files, so it stays.
- `README.md` and `handover.md` mention a local tag `frozen-gru-study-20261021` that is not on GitHub. It may exist on the machine that made it, so the text was not changed.
- The empty `.gru_univar-*` folder left by the stopped ROCm run in `outputs/` is local, untracked data and was not deleted.
