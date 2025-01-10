from collections.abc import Iterable


def coordinate_to_bbox(coordinate, *dims):
    assert len(coordinate) == 2
    if (len(dims) == 1) and isinstance(dims[0], Iterable):
        return [*coordinate, *dims[0]]
    elif isinstance(dims[0], int):
        return [*coordinate, *dims]
