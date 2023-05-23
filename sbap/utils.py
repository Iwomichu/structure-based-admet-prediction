from typing import TypeVar, Sequence, Generator

_T = TypeVar("_T")


def batched(iterable: Sequence[_T], n=1) -> Generator[Sequence[_T], None, None]:
    iterable_length = len(iterable)
    for ndx in range(0, iterable_length, n):
        yield iterable[ndx:min(ndx + n, iterable_length)]
