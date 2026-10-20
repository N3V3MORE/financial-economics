"""Refresh the supplemental Excel return snapshot from its existing Input sheet.

Run from the repository root with:
    python scripts/refresh_supplement.py --in-place

Only the Returns sheet, explanatory cells on three price-summary sheets, and
the calculation-chain entries for replaced formulas change. All other XLSX
package parts, including the raw inputs and pivots, retain their uncompressed
contents. The results are snapshots; editing Input alone does not recalculate
them.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta
import math
import os
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from zipfile import ZipFile


MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"m": MAIN}
TICKERS = ("ARKK", "DBC", "GLD", "IWF", "QQQ", "SCHG", "SPY", "VNQ", "VUG", "XLE", "XLK", "XLY")
YEARS = range(2000, 2026)
CELL = re.compile(rb'<c\b[^>]*?\br="([A-Z]+[0-9]+)"[^>]*?(?:/>|>.*?</c>)', re.DOTALL)


def sheet_paths(archive: ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        item.attrib["Id"]: item.attrib["Target"]
        for item in relationships.findall(f"{{{PACKAGE_REL}}}Relationship")
    }
    result = {}
    for sheet in workbook.findall("m:sheets/m:sheet", NS):
        target = targets[sheet.attrib[f"{{{OFFICE_REL}}}id"]]
        result[sheet.attrib["name"]] = (
            target.lstrip("/") if target.startswith("/") else "xl/" + target
        )
    return result


def shared_strings(archive: ZipFile) -> list[str]:
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(node.text or "" for node in item.iter(f"{{{MAIN}}}t"))
        for item in root.findall("m:si", NS)
    ]


def sheet_id(archive: ZipFile, name: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    for sheet in workbook.findall("m:sheets/m:sheet", NS):
        if sheet.attrib["name"] == name:
            return sheet.attrib["sheetId"]
    raise ValueError(f"Missing sheet: {name}")


def remove_cells_from_calc_chain(
    xml: bytes, target_sheet_id: str, replaced_addresses: set[str]
) -> bytes:
    root = ET.fromstring(xml)
    current_sheet_id = None
    for item in list(root):
        current_sheet_id = item.attrib.get("i", current_sheet_id)
        if current_sheet_id == target_sheet_id and item.attrib["r"] in replaced_addresses:
            root.remove(item)
        else:
            item.attrib["i"] = current_sheet_id
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def raw_returns(archive: ZipFile, input_path: str) -> tuple[dict[str, dict[date, float | None]], int]:
    strings = shared_strings(archive)
    seen: dict[tuple[int, date], tuple] = {}
    series: dict[str, dict[date, float | None]] = defaultdict(dict)
    ticker_ids: dict[str, int] = {}
    duplicate_count = 0

    with archive.open(input_path) as stream:
        for _, row in ET.iterparse(stream, events=("end",)):
            if row.tag != f"{{{MAIN}}}row":
                continue
            if int(row.attrib["r"]) == 1:
                headers = {}
                for cell in row.findall("m:c", NS):
                    column = re.match(r"[A-Z]+", cell.attrib["r"]).group()
                    value = cell.findtext("m:v", default=None, namespaces=NS)
                    headers[column] = strings[int(value)] if cell.attrib.get("t") == "s" else value
                expected = {"A": "PERMNO", "C": "Ticker", "E": "DlyCalDt", "K": "DlyRet"}
                if any(headers.get(column) != label for column, label in expected.items()):
                    raise ValueError("Input columns A/C/E/K do not match expected CRSP fields")
                row.clear()
                continue
            cells = {}
            fingerprint = []
            for cell in row.findall("m:c", NS):
                column = re.match(r"[A-Z]+", cell.attrib["r"]).group()
                value = cell.findtext("m:v", default=None, namespaces=NS)
                kind = cell.attrib.get("t")
                fingerprint.append((column, kind, value))
                cells[column] = (kind, value)
            row.clear()
            if "A" not in cells:
                continue
            permno = int(cells["A"][1])
            ticker = strings[int(cells["C"][1])]
            serial = float(cells["E"][1])
            observation_date = (datetime(1899, 12, 30) + timedelta(days=serial)).date()
            previous_id = ticker_ids.setdefault(ticker, permno)
            if previous_id != permno:
                raise ValueError(f"Ticker {ticker} has multiple PERMNOs")
            key = (permno, observation_date)
            signature = tuple(fingerprint)
            if key in seen:
                if seen[key] != signature:
                    raise ValueError(f"Conflicting rows for {ticker} on {observation_date}")
                duplicate_count += 1
                continue
            seen[key] = signature
            field = cells.get("K")
            daily_return = None if field is None or field[1] is None else float(field[1])
            series[ticker][observation_date] = daily_return

    if set(series) != set(TICKERS):
        raise ValueError(f"Unexpected ticker set: {sorted(series)}")
    if not series["SPY"]:
        raise ValueError("SPY trading calendar is unavailable")
    return series, duplicate_count


def period_return(
    ticker_dates: dict[date, float | None],
    spy_dates: dict[date, float | None],
    year: int,
    month: int | None = None,
    quarter: int | None = None,
) -> float | None:
    def selected(day: date) -> bool:
        return day.year == year and (month is None or day.month == month) and (
            quarter is None or (day.month - 1) // 3 + 1 == quarter
        )

    calendar = {day for day in spy_dates if selected(day)}
    observed = {day for day in ticker_dates if selected(day)}
    if not calendar or observed != calendar:
        return None
    returns = [ticker_dates[day] for day in sorted(calendar)]
    if any(value is None or not math.isfinite(value) or value <= -1 for value in returns):
        return None
    return math.prod(1 + value for value in returns) - 1


def column_name(index: int) -> str:
    result = ""
    while index:
        index, digit = divmod(index - 1, 26)
        result = chr(65 + digit) + result
    return result


def snapshot_cells(series: dict[str, dict[date, float | None]]) -> dict[str, float | None]:
    result = {}
    for ticker_number, ticker in enumerate(TICKERS):
        monthly_row = 2 + ticker_number * 13
        quarterly_row = 160 + ticker_number * 5
        for year in YEARS:
            column = column_name(year - 2000 + 3)
            annual = period_return(series[ticker], series["SPY"], year)
            result[f"{column}{monthly_row}"] = annual
            result[f"{column}{quarterly_row}"] = annual
            for month in range(1, 13):
                result[f"{column}{monthly_row + month}"] = period_return(
                    series[ticker], series["SPY"], year, month=month
                )
            for quarter in range(1, 5):
                result[f"{column}{quarterly_row + quarter}"] = period_return(
                    series[ticker], series["SPY"], year, quarter=quarter
                )
    return result


def patch_cells(xml: bytes, replacements: dict[str, float | str | None]) -> bytes:
    remaining = set(replacements)

    def content(opening: bytes, value: float | str | None) -> bytes:
        opening = opening.rstrip(b"/")
        opening = re.sub(rb'\s+t="[^"]*"', b"", opening)
        if value is None or isinstance(value, str):
            label = "n.a." if value is None else value
            return opening + b' t="inlineStr"><is><t>' + escape(label).encode() + b"</t></is></c>"
        return opening + b"><v>" + format(value, ".17g").encode("ascii") + b"</v></c>"

    def replace(match: re.Match[bytes]) -> bytes:
        address = match.group(1).decode("ascii")
        if address not in remaining:
            return match.group(0)
        remaining.remove(address)
        original = match.group(0)
        return content(original.split(b">", 1)[0], replacements[address])

    updated = CELL.sub(replace, xml)
    if remaining:
        by_row: dict[int, list[str]] = defaultdict(list)
        for address in remaining:
            by_row[int(re.search(r"[0-9]+", address).group())].append(address)

        def insert(row_match: re.Match[bytes]) -> bytes:
            row_number = int(row_match.group(1))
            if row_number not in by_row:
                return row_match.group(0)
            row_xml = row_match.group(0)
            for address in sorted(by_row.pop(row_number), key=lambda a: column_number(a), reverse=True):
                existing = list(CELL.finditer(row_xml))
                next_cell = next(
                    (cell for cell in existing if column_number(cell.group(1).decode()) > column_number(address)),
                    None,
                )
                position = next_cell.start() if next_cell else row_xml.index(b"</row>")
                style_match = re.search(rb'\bs="([0-9]+)"', next_cell.group(0)) if next_cell else None
                opening = f'<c r="{address}"'.encode()
                if style_match:
                    opening += b' s="' + style_match.group(1) + b'"'
                row_xml = row_xml[:position] + content(opening, replacements[address]) + row_xml[position:]
                remaining.remove(address)
            return row_xml

        updated = re.sub(rb'<row\b[^>]*\br="([0-9]+)"[^>]*>.*?</row>', insert, updated, flags=re.DOTALL)
        if by_row or remaining:
            raise ValueError(f"Missing expected rows or cells: {sorted(remaining)[:8]}")
    return updated


def column_number(address: str) -> int:
    result = 0
    for letter in re.match(r"[A-Z]+", address).group():
        result = result * 26 + ord(letter) - 64
    return result


def append_note(xml: bytes, row_number: int, note: str) -> bytes:
    if re.search(rb'<row\b[^>]*\br="' + str(row_number).encode() + rb'"', xml):
        return patch_cells(xml, {f"A{row_number}": note})
    cell = (
        f'<row r="{row_number}"><c r="A{row_number}" t="inlineStr"><is><t>'
        f'{escape(note)}</t></is></c></row>'
    ).encode("utf-8")
    if b"</sheetData>" not in xml:
        raise ValueError("Worksheet is missing sheetData")
    updated = xml.replace(b"</sheetData>", cell + b"</sheetData>", 1)
    return re.sub(
        rb'(<dimension ref="[A-Z]+1:)([A-Z]+)[0-9]+(")',
        lambda match: match.group(1) + match.group(2) + str(row_number).encode() + match.group(3),
        updated,
        count=1,
    )


def refresh(workbook_path: Path) -> tuple[int, int, int]:
    temporary_path = None
    with ZipFile(workbook_path) as source:
        paths = sheet_paths(source)
        required = {"Input", "Returns", "Yearly Price", "Quarterly Price", "Price Changes"}
        if not required.issubset(paths):
            raise ValueError("Workbook is missing an expected sheet")
        series, duplicate_count = raw_returns(source, paths["Input"])
        values = snapshot_cells(series)
        updated_sheets = {}
        returns_path = paths["Returns"]
        returns_xml = source.read(returns_path)
        returns_xml = patch_cells(returns_xml, values | {"A1": "Snapshot"})
        returns_xml = append_note(
            returns_xml,
            220,
            "Returns snapshot: compound Input DlyRet by calendar year, month, and quarter after removing exact duplicate PERMNO/date rows.",
        )
        returns_xml = append_note(
            returns_xml,
            221,
            "n.a. means the ticker lacks a complete SPY trading calendar or a required daily return. Missing periods are not zero returns.",
        )
        returns_xml = append_note(
            returns_xml,
            222,
            "Figures do not update when Input changes. Refresh with: python scripts/refresh_supplement.py --in-place. Scope: 12 Input ETFs; TLT and VIG are absent.",
        )
        updated_sheets[returns_path] = returns_xml
        notes = {
            "Yearly Price": (161, "Sums of daily prices are descriptive totals, not investment returns."),
            "Quarterly Price": (64, "Sums of daily prices are descriptive totals, not investment returns."),
            "Price Changes": (316, "Differences between sums of daily prices are descriptive, not investment gains or losses."),
        }
        for sheet_name, (row_number, note) in notes.items():
            path = paths[sheet_name]
            updated_sheets[path] = append_note(source.read(path), row_number, note)
        if "xl/calcChain.xml" in source.namelist():
            updated_sheets["xl/calcChain.xml"] = remove_cells_from_calc_chain(
                source.read("xl/calcChain.xml"), sheet_id(source, "Returns"), set(values)
            )

        with tempfile.NamedTemporaryFile(
            dir=workbook_path.parent, prefix=".return-refresh-", suffix=".xlsx", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with ZipFile(temporary_path, "w") as destination:
                destination.comment = source.comment
                for item in source.infolist():
                    destination.writestr(item, updated_sheets.get(item.filename, source.read(item)))
            with ZipFile(temporary_path) as check:
                if check.testzip() is not None:
                    raise ValueError("Updated workbook failed ZIP validation")
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    try:
        os.replace(temporary_path, workbook_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    available = sum(value is not None for value in values.values())
    return duplicate_count, available, len(values) - available


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbook", type=Path,
        default=Path(__file__).resolve().parents[1] / "Data V2.xlsx",
    )
    parser.add_argument("--in-place", action="store_true", help="replace the selected workbook")
    args = parser.parse_args()
    if not args.in_place:
        parser.error("Specify --in-place to refresh the selected workbook")
    workbook = args.workbook.resolve(strict=True)
    if workbook.suffix.lower() != ".xlsx" or not workbook.is_file():
        parser.error("--workbook must be an existing .xlsx file")
    duplicates, available, unavailable = refresh(workbook)
    print(
        f"Refreshed {workbook.name}: {duplicates} exact duplicate rows skipped; "
        f"{available} period results; {unavailable} n.a."
    )


if __name__ == "__main__":
    main()
