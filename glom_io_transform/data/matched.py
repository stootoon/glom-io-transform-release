"""The matched input/output glomerulus pairs, as one object.

Tobias identified pairs of glomeruli imaged in both the input and the output
datasets, and exported them as three CSVs: one of pair metadata, and one each
of the trial-averaged, [0,1]-normalised input and output responses. This module
reads all three and returns a single xarray Dataset, so the pairing, the
metadata and the responses can be indexed together.

    from glom_io_transform.data.matched import load_matched, matched_table
    m = load_matched()
    m.sel(match=3)            # everything about one pair
    matched_table(m)          # the pairing as a DataFrame, for reading

The response columns keep the dim name 'csv_column' rather than 'odour',
because the values are not in the order the exported header names them.
"""
from .common import WARN
from .odours import get_data_file

# The leading columns of the response CSVs repeat the pair metadata; everything
# after them is a response.
META_COLUMNS = ["match_id", "input_row", "output_row", "input_exp", "output_exp",
                "input_local_roi", "output_local_roi", "distance", "correlation"]

# Which pair a row is becomes a coordinate; what was measured becomes a variable.
_COORDS = ["input_row", "output_row", "input_exp", "output_exp",
           "input_local_roi", "output_local_roi",
           "input_x", "input_y", "output_x", "output_y"]


def load_matched():
    """The matched pairs as an xarray Dataset.

    Dimensions:
        match       the pairs, labelled by match_id
        csv_column  the response columns, labelled by the exported header

    Variables:
        input, output   (match, csv_column)   the responses
        distance        (match,)              between the two ROIs
        correlation     (match,)              as reported in the export
    """
    import pandas as pd
    import xarray as xr

    meta = pd.read_csv(get_data_file("matched_roi_pairs_metadata.csv"))
    csvs = {side: pd.read_csv(get_data_file(f"matched_roi_{side}_trial_averaged_normalized.csv"))
            for side in ("input", "output")}

    columns = [c for c in csvs["input"].columns if c not in META_COLUMNS]
    assert columns == [c for c in csvs["output"].columns if c not in META_COLUMNS], \
        "The input and output response CSVs do not have the same columns."

    # The three files are separate exports and their match_id columns disagree
    # -- the metadata numbers the pairs from 0, the response CSVs from 1 -- so
    # line them up on the pair of row indices, which identify a pair however it
    # was numbered, and keep the metadata's numbering.
    key = ["input_row", "output_row"]
    meta = meta.sort_values(key).set_index("match_id")
    frames = {side: df.sort_values(key) for side, df in csvs.items()}
    for side, df in frames.items():
        assert df[key].values.tolist() == meta[key].values.tolist(), \
            f"The {side} response CSV covers different pairs than the metadata."
        for c in ("input_exp", "output_exp", "input_local_roi", "output_local_roi", "distance"):
            if c in df and not (df[c].values == meta[c].values).all():
                WARN(f"The {side} response CSV disagrees with the metadata on {c!r}.")

    ds = xr.Dataset(
        data_vars={"input":       (("match", "csv_column"), frames["input"][columns].values),
                   "output":      (("match", "csv_column"), frames["output"][columns].values),
                   "distance":    ("match", meta["distance"].values),
                   "correlation": ("match", meta["correlation"].values)},
        coords={"match": meta.index.values, "csv_column": columns,
                **{c: ("match", meta[c].values) for c in _COORDS}},
        attrs={"source": "matched_roi_*.csv",
               "csv_column_note": "the header labels are not the order the values are in"},
    )
    return ds.sortby("match")


def matched_table(ds):
    """The pairing and its metadata as a DataFrame, for reading rather than computing."""
    import pandas as pd
    cols = [c for c in _COORDS if c.startswith("input")] + \
           [c for c in _COORDS if c.startswith("output")]
    df = pd.DataFrame({c: ds[c].values for c in cols},
                      index=pd.Index(ds.match.values, name="match_id"))
    df["distance"]    = ds.distance.values
    df["correlation"] = ds.correlation.values
    return df
