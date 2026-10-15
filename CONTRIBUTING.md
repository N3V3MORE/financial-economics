# Contributing

Accept your collaborator invitation before cloning the repository or pushing changes.

1. Open an issue describing the question or change you want to work on.
2. Update your local `main` branch, then create a branch for your task:

   ```sh
   git switch main
   git pull --ff-only
   git switch -c your-task-name
   ```

3. Keep the change focused. Include enough detail for someone else to reproduce calculations and understand the results.
4. Check your work, commit it, and push your branch:

   ```sh
   git add path/to/your-changed-file
   git commit -m "Describe your change"
   git push -u origin your-task-name
   ```

5. Open a pull request on GitHub. Explain what changed, why, and how you checked it. Ask another contributor to review it before merging.

Do not commit passwords, API keys, or data you do not have permission to share with the collaborators.
