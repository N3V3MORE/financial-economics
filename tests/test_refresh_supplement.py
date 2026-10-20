from datetime import date
import importlib.util
from pathlib import Path
import unittest


script_path = Path(__file__).resolve().parents[1] / "scripts" / "refresh_supplement.py"
spec = importlib.util.spec_from_file_location("refresh_supplement", script_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
append_note = module.append_note
patch_cells = module.patch_cells
period_return = module.period_return
remove_cells_from_calc_chain = module.remove_cells_from_calc_chain


class ReturnSnapshotTests(unittest.TestCase):
    def test_complete_period_compounds_daily_returns(self):
        calendar = {date(2022, 1, 3): 0.0, date(2022, 1, 4): 0.0}
        returns = {date(2022, 1, 3): 0.10, date(2022, 1, 4): -0.05}
        self.assertAlmostEqual(period_return(returns, calendar, 2022), 0.045)

    def test_missing_day_or_return_is_unavailable(self):
        calendar = {date(2022, 1, 3): 0.0, date(2022, 1, 4): 0.0}
        self.assertIsNone(period_return({date(2022, 1, 3): 0.1}, calendar, 2022))
        self.assertIsNone(period_return({date(2022, 1, 3): 0.1, date(2022, 1, 4): None}, calendar, 2022))

    def test_xml_patch_replaces_error_and_preserves_style(self):
        original = b'<c r="H54" s="33" t="e"><f>1/0</f><v>#DIV/0!</v></c>'
        changed = patch_cells(original, {"H54": None})
        self.assertEqual(changed, b'<c r="H54" s="33" t="inlineStr"><is><t>n.a.</t></is></c>')

    def test_note_refresh_is_repeatable(self):
        original = b'<worksheet><dimension ref="A1:B2"/><sheetData><row r="2"/></sheetData></worksheet>'
        once = append_note(original, 3, "first")
        twice = append_note(once, 3, "updated")
        self.assertIn(b'<dimension ref="A1:B3"', twice)
        self.assertIn(b'<t>updated</t>', twice)
        self.assertNotIn(b'<t>first</t>', twice)

    def test_xml_patch_inserts_missing_cell_in_column_order(self):
        original = b'<row r="2"><c r="B2"/><c r="D2" s="33"><v>0</v></c></row>'
        changed = patch_cells(original, {"C2": 0.1})
        self.assertIn(b'<c r="C2" s="33"><v>0.10000000000000001</v></c>', changed)
        self.assertLess(changed.index(b'r="B2"'), changed.index(b'r="C2"'))
        self.assertLess(changed.index(b'r="C2"'), changed.index(b'r="D2"'))

    def test_calc_chain_removes_return_formulas_and_keeps_inherited_ids(self):
        original = (
            b'<calcChain xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            b'<c r="D2" i="5"/><c r="E2" i="8"/><c r="F2"/>'
            b'<c r="A1"/><c r="G2" i="9"/>'
            b'</calcChain>'
        )
        changed = remove_cells_from_calc_chain(original, "8", {"E2", "F2"})
        self.assertNotIn(b'r="E2"', changed)
        self.assertNotIn(b'r="F2"', changed)
        self.assertIn(b'r="D2"', changed)
        self.assertIn(b'r="A1" i="8"', changed)
        self.assertIn(b'r="G2"', changed)


if __name__ == "__main__":
    unittest.main()
