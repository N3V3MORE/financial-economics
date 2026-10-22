import pandas as pd
import pytest
from financial_economics.settings.study import Study
from financial_economics.datasets.returns import compound_months
from financial_economics.datasets.sources.crsp import validate_monthly_extract, WRDSError


def test_complete_months_and_monthly_reconciliation():
    dates = pd.bdate_range("2020-01-01", "2020-02-29")
    daily = pd.DataFrame({"A": .001, "B": -.001}, index=dates)
    study = Study("test", "2020-01-01", "2020-02-29", "A", {"A": "A", "B": "B"})
    monthly = compound_months(daily, dates, study.start, study.end)
    assert monthly.A.iloc[0] == pytest.approx(1.001 ** 23 - 1)
    with pytest.raises(ValueError, match="every expected"):
        compound_months(daily.iloc[1:], dates, study.start, study.end)
    with pytest.raises(ValueError, match="complete"):
        compound_months(daily, dates, "2020-01-02", study.end)
    raw = monthly.rename(columns={"A": 1, "B": 2}).rename_axis(columns="permno").stack().rename("total_return").reset_index()
    raw = raw.rename(columns={"Date": "date"})
    raw["return_flag"] = "CR"
    raw = raw[["permno", "date", "total_return", "return_flag"]]
    assert validate_monthly_extract(daily, raw, {"A": 1, "B": 2}, dates, study)["status"] == "passed"
    bad_flag = raw.copy()
    bad_flag.loc[0, "return_flag"] = "BAD"
    with pytest.raises(WRDSError, match="FLAG_INVALID"):
        validate_monthly_extract(daily, bad_flag, {"A": 1, "B": 2}, dates, study)
    raw.loc[0, "total_return"] += .00011
    with pytest.raises(WRDSError, match="RECONCILIATION"):
        validate_monthly_extract(daily, raw, {"A": 1, "B": 2}, dates, study)
