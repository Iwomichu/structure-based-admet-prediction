import logging
import re
from pathlib import Path
from typing import Optional, TypedDict, Generator, TextIO


class ChemblSdfRecord(TypedDict):
    mol: str
    cdId: int
    standardValue: float


class ChemblSdfReader:
    def __init__(self, logging_level: int = logging.INFO) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging_level)

    def _non_empty_line_or_none(self, file: TextIO, max_empty_lines_distance: int = 5) -> Optional[str]:
        empty_lines = 0
        line = file.readline()
        self.logger.debug(f"{line=}")
        while line.strip() == "":
            empty_lines += 1
            if empty_lines > max_empty_lines_distance:
                return None
            line = file.readline()
            self.logger.debug(f"{line=}")
        return line

    def chembl_file_record_generator(
            self,
            path: Path,
            max_empty_lines_distance: int = 5,
    ) -> Generator[ChemblSdfRecord, None, None]:
        with open(path, mode='r', encoding='utf-8') as file:
            while True:
                line = self._non_empty_line_or_none(file, max_empty_lines_distance)
                if line is None:
                    return
                mol = ""
                while not re.match(r">.* <CdId>", line):
                    mol += line
                    line = file.readline()
                    self.logger.debug(f"{line=}")
                cd_id = int(file.readline())
                self.logger.debug(f"{cd_id=}")
                while not re.match(r">.* <Standard Value>", file.readline()):
                    pass
                standard_value = float(file.readline())
                self.logger.debug(f"{standard_value=}")
                while not re.match(r".*\$\$\$\$", file.readline()):
                    pass
                record = ChemblSdfRecord(mol=mol.strip(), cdId=cd_id, standardValue=standard_value)
                yield record

    def parse(self, path: Path) -> list[ChemblSdfRecord]:
        return [record for record in self.chembl_file_record_generator(path)]
