import pathlib
import unittest

from sbap.sdf import ChemblSdfReader, ChemblSdfRecord

PARENT_DIR = pathlib.Path(__file__).parent.resolve()
with open(PARENT_DIR / "resources" / "result_mol_1.txt") as f:
    MOL_1 = f.read()

with open(PARENT_DIR / "resources" / "result_mol_2.txt") as f:
    MOL_2 = f.read()


class SdfReaderTestCase(unittest.TestCase):
    def test_something(self):
        reader = ChemblSdfReader()

        expected = [
            ChemblSdfRecord(
                mol=MOL_1,
                cdId=1,
                standardValue=100.00,
            ),
            ChemblSdfRecord(
                mol=MOL_2,
                cdId=2,
                standardValue=9900.00,
            ),
        ]
        result = reader.parse(
            PARENT_DIR / "resources" / "test_ligand.sdf",
        )

        self.assertListEqual(expected, result)  # add assertion here


if __name__ == '__main__':
    unittest.main()
