import re
from pathlib import Path
from typing import Optional, TypedDict, Generator, TextIO


class ChemblSdfRecord(TypedDict):
    mol: str
    cdId: int
    standardValue: float


def _non_empty_line_or_none(file: TextIO, max_empty_lines_distance: int = 5) -> Optional[str]:
    empty_lines = 0
    line = file.readline()
    while line.strip() == "":
        empty_lines += 1
        if empty_lines > max_empty_lines_distance:
            return None
        line = file.readline()
    return line


def chembl_file_record_generator(
        path: Path,
        max_empty_lines_distance: int = 5,
) -> Generator[ChemblSdfRecord, None, None]:
    with open(path, mode='r', encoding='utf-8') as file:
        while True:
            line = _non_empty_line_or_none(file, max_empty_lines_distance)
            if line is None:
                return
            mol = ""
            while not re.match(r">.* <CdId>", line):
                mol += line
                line = file.readline()
                print(f"{line=}")
            cd_id = int(file.readline())
            print(f"{cd_id=}")
            while not re.match(r">.* <Standard Value>", file.readline()):
                pass
            standard_value = float(file.readline())
            print(f"{standard_value=}")
            while not re.match(r".*\$\$\$\$", file.readline()):
                pass
            record = ChemblSdfRecord(mol=mol.strip(), cdId=cd_id, standardValue=standard_value)
            yield record


class ChemblSdfReader:
    def __init__(self) -> None:
        pass

    def parse(self, path: Path) -> list[ChemblSdfRecord]:
        return [record for record in chembl_file_record_generator(path)]
