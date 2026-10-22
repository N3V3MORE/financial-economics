# Contributing

Accept your collaborator invitation before cloning the repository or pushing changes.

1. Open an issue describing the question or change you want to work on.
2. Update your local `main` branch, then create a branch for your task:

   ```sh
   git switch main
   git pull --ff-only
   git switch -c your-task-name
   ```

3. Install the locked environment with `uv sync --frozen --extra reconstruction`. The extra installs `arch`, which the GARCH model tests need. Keep the change focused, and explain the economic meaning of any calculation change.
4. Run `uv run --frozen ruff check` and then `uv run --frozen --extra reconstruction pytest -q -W error`. These are the checks CI runs. Add a small synthetic regression test for changed calculation or data-validation behavior. Tests must not call live finance providers.
5. Commit your work and push your branch:

   ```sh
   git add path/to/your-changed-file
   git commit -m "Describe your change"
   git push -u origin your-task-name
   ```

6. Open a pull request on GitHub. Explain what changed, why, and how you checked it. Ask another contributor to review it before merging.

Change the study universe or sample explicitly in `configs/study.toml`, and document the reason. Fetch a new snapshot for changed settings instead of editing retained data. Keep provider data and generated outputs out of Git; share the commands and interpretation needed to reproduce them.

Do not commit passwords, API keys, or data you do not have permission to share with the collaborators.
