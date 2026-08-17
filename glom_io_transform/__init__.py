"""glom_io_transform.

Sets xarray's arithmetic_join to "exact" for the whole process: operations on
labelled arrays whose coordinates disagree (e.g. two sets of odour responses in
different orders, or over different odour sets) raise instead of being silently
aligned or inner-joined. Mismatched labels are a bug in this codebase, not
something to paper over.
"""
try:
    import xarray as _xr
    _xr.set_options(arithmetic_join="exact")
except ImportError:  # xarray is only needed for the data layer
    pass
