def get_indices_for_split(splits, split_name, length=None):
    split_idx = splits[split_name]

    # Case 1: slice
    if isinstance(split_idx, slice):
        idx = list(range(split_idx.start, split_idx.stop))

    # Case 2: tuple (start, end)
    elif isinstance(split_idx, tuple) and len(split_idx) == 2:
        start, end = split_idx
        idx = list(range(start, end))

    # Case 3: already list/array
    else:
        idx = list(split_idx)

    # If we only want the last N indices
    if length is not None:
        idx = idx[-length:]

    return idx
