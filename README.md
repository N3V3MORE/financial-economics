# Financial economics: ETF protection and volatility timing

Can a stock investor protect against market falls better than by holding less
stock? This project starts with an eight-ETF diversification study, then compares
ETF additions with cash and studies volatility-based stock allocation. It retains
the paper replication, realistic trading tests, methodology revision and later
Risk Lab trial as distinct experiments.

The saved findings are retrospective. In the matched 2011–2020 revision, lower
forecast error did not establish better portfolio performance: no comparison
survived the joint 27-test adjustment. The October 22 Risk Lab trial also found
smaller worst falls in 2022–2025 without a statistically established improvement
over static stock/cash mixes; long Treasuries worsened its stress-period results.
The estimated Oxford-like substitute failed the required daily accuracy floor
and remains diagnostic only. These results do not establish a reliable safe haven
or a profitable forecasting edge. Read [Final study](docs/final-study.md),
[Results explained](docs/results-explained.md) and the dated [Handover](handover.md)
for the evidence and limits.

## Run the original study

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from
the project root:

```sh
uv sync --frozen
uv run fecon download
uv run fecon analyze
uv run fecon stress
uv run fecon compare
uv run fecon explain
uv run fecon page
```

Skip downloads when the matching snapshot already exists. Phases 1–4 write
`outputs/phase1/summary.md` through `outputs/phase4/summary.md`; the combined page
is `outputs/page/index.html`. New snapshots and runs need new destination names.

The retained main protection study uses verified WRDS/CRSP returns:

```sh
uv run fecon --config configs/protection.toml protection --data-dir data/protection-wrds-20261020 --output-dir outputs/protection-main-final
```

That command requires the matching local snapshot and an unused output directory.
See [Commands and study reference](docs/commands.md) for its original download,
Windows login and independent verification commands, and the longer sample.

## Run the current matched study

This existing staged command uses Risk Lab volatility with the retained macro and
ETF-return snapshots. It does not recreate the archived Oxford run. In PowerShell,
with those inputs present and a new output directory:

```powershell
$revisedArgs = @('--data-dir', 'data/paper-wrds-eia-20261021', '--returns-dir', 'data/protection-long-wrds-20261021', '--risklab-dir', 'data/risklab-spy-20261022', '--output-dir', 'outputs/revised-study-new')
foreach ($stage in @('freeze', 'classical', 'gru_univar_log', 'gru_multivar_log', 'evaluate')) {
    uv run --extra prediction --extra reconstruction fecon paper-revised $stage @revisedArgs
    if ($LASTEXITCODE -ne 0) { break }
}
```

GRU training requires a compatible GPU. NVIDIA uses the `prediction` extra; the
supported Windows AMD setup uses `prediction-rocm` instead. The extras are mutually
exclusive. Full setup, archived and recent study commands, forecast reuse,
trading schedules, figures and post hoc checks remain in
[Commands and study reference](docs/commands.md).

The later-year Risk Lab trial runs as its own staged `fecon` subcommand. Inspect its
existing flags before running it against local sources:

```sh
fecon risklab-study --help
```

## Code and development

Implementation now follows seven responsibility groups: `datasets`, `settings`,
`models`, `portfolio`, `evaluation`, `studies` and `reporting`. Every study and
source tool runs through `fecon` (or `python -m financial_economics`); the staged
studies that were once run as separate modules are now `fecon` subcommands with the
same options. No methods, features or studies were removed. Configurations remain
in `configs/`; existing snapshot paths, output paths and formats stay unchanged.
[Project structure](docs/structure.md) maps every former flat module to its
grouped implementation.

Use [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and locked
checks. Keep downloaded data and generated results local in the ignored `data/`
and `outputs/` directories. Licensed WRDS observations must not be uploaded or
redistributed. Tests use synthetic inputs and must not call live finance providers.
