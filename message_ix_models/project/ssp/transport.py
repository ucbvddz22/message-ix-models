"""Postprocess aviation emissions for SSP 2024."""

import re
from typing import TYPE_CHECKING, Dict, Hashable, List

import genno
import xarray as xr

from message_ix_models.tools.iamc import iamc_like_data_for_query

if TYPE_CHECKING:
    import pathlib

    from genno.types import AnyQuantity

# TODO Retrieve from a package data code list
EMISSION = "BC CH4 CO CO2 N2O NH3 NOx OC Sulfur VOC".split()


def aviation_share() -> "AnyQuantity":
    """Return (dummy) data for the share of aviation in emissions.

    Returns
    -------
    genno.Quantity
        with dimensions :math:`(n, e, y)`.
    """
    return (
        genno.Quantity(0.2, units="dimensionless")
        .expand_dims({"e": EMISSION})  # Full list
        .expand_dims({"n": ["CHN", "NAM"]})  # Test data specimen
        .expand_dims({"y": list(range(1990, 2110 + 1))})  # Crudely cover the full list
    )


def finalize(
    q_all: "AnyQuantity",
    q_update: "AnyQuantity",
    model_name: str,
    scenario_name: str,
    path_out: "pathlib.Path",
) -> None:
    """Finalize output.

    1. Reattach "Model" and "Scenario" labels.
    2. Reassemble the "Variable" dimension/coords of `q_update`; drop "e" and "t".
    3. Convert both `q_all` and `q_update` to :class:`pandas.Series`; update the former
       with the contents of the latter.
    4. Adjust to IAMC ‘wide’ structure and write to `path_out`.

    Parameters
    ----------
    q_all :
        All data.
    q_update :
        Revised data to overwrite corresponding values in `q_all`.
    """

    def _expand(qty):
        return qty.expand_dims(
            {"Model": [model_name], "Scenario": [scenario_name]}
        ).rename({"n": "Region", "UNIT": "Unit", "VARIABLE": "Variable"})

    s_all = q_all.pipe(_expand).to_series()

    s_all.update(
        q_update.pipe(_expand)
        .to_frame()
        .reset_index()
        .assign(
            Variable=lambda df: (
                "Emission|" + df["e"] + "|Energy|Demand|Transportation|" + df["t"]
            ).str.replace("|_T", "")
        )
        .drop(["e", "t"], axis=1)
        .set_index(s_all.index.names)[0]
    )

    (
        s_all.unstack("y")
        .reorder_levels(["Model", "Scenario", "Region", "Variable", "Unit"])
        .reset_index()
        .to_csv(path_out, index=False)
    )


def extract_dims(
    qty: "AnyQuantity", dim_expr: dict, *, drop: bool = True, fillna: str = "_T"
) -> "AnyQuantity":
    """Extract dimensions from IAMC-like ‘variable’ names using regular expressions."""
    import pandas as pd

    dims = list(qty.dims)

    dfs = [qty.to_frame().reset_index()]
    for dim, expr in dim_expr.items():
        pattern = re.compile(expr)
        dfs.append(dfs[0][dim].str.extract(pattern).fillna(fillna))
        dims.extend(pattern.groupindex)
        if drop:
            dims.remove(dim)

    return genno.Quantity(pd.concat(dfs, axis=1).set_index(dims)["value"])


def extract_dims1(qty: "AnyQuantity", dim: dict) -> "AnyQuantity":  # pragma: no cover
    """Extract dimensions from IAMC-like ‘variable’ names expressions.

    .. note:: This incomplete, non-working version uses :mod:`xarray` semantics.
    """
    from collections import defaultdict

    result = qty
    for d0, expr in dim.items():
        d0_new = f"{d0}_new"
        pattern = re.compile(expr)

        indexers: Dict[Hashable, List[Hashable]] = {g: [] for g in pattern.groupindex}
        indexers[d0_new] = []

        coords = qty.coords[d0].data.astype(str)
        for coord in coords:
            if match := pattern.match(coord):
                groupdict = match.groupdict()
                coord_new = coord[match.span()[1] :]
            else:
                groupdict = defaultdict(None)
                coord_new = coord

            for g in pattern.groupindex:
                indexers[g].append(groupdict[g])
            indexers[d0_new].append(coord_new)

        for d1, labels in indexers.items():
            i2 = {d0: xr.DataArray(coords, coords={d1: labels})}
            result = result.sel(i2)

    return result


def select_re(qty: "AnyQuantity", indexers: dict) -> "AnyQuantity":
    """Select using regular expressions for each dimension."""
    new_indexers = dict()
    for dim, expr in indexers.items():
        new_indexers[dim] = list(
            map(str, filter(re.compile(expr).match, qty.coords[dim].data.astype(str)))
        )
    return qty.sel(new_indexers)


EXPR = r"^Emissions\|(?P<e>[^\|]+)\|Energy\|Demand\|Transportation(?:\|(?P<t>.*))?$"


def main(path_in: "pathlib.Path", path_out: "pathlib.Path"):
    """Postprocess aviation emissions for SSP 2024.

    Parameters
    ----------
    path_in :
        Input data path.
    path_out :
        Output data path.
    """
    import pandas as pd

    # Shorthand
    e_t = ("e", "t")
    t = "t"
    k_input = genno.Key("input", ("n", "y", "VARIABLE", "UNIT"))
    k = genno.KeySeq("result", ("n", "y", "UNIT") + e_t)

    c = genno.Computer()

    # Read the data from `path`
    c.add(
        k_input,
        iamc_like_data_for_query,
        path=path_in,
        query="Model != ''",
        unique="MODEL SCENARIO",
    )

    # Peek at `path` to identify the model and scenario names
    df = pd.read_csv(path_in, nrows=1)
    c.add("model name", genno.quote(df["Model"].iloc[0]))
    c.add("scenario name", genno.quote(df["Scenario"].iloc[0]))

    # Filter on "VARIABLE"
    c.add(k[0] / e_t, select_re, k_input, indexers={"VARIABLE": EXPR})

    # Extract the "e" and "t" dimensions from "VARIABLE"
    c.add(k[1], extract_dims, k[0] / e_t, dim_expr={"VARIABLE": EXPR})

    # Select the total
    c.add(k[2] / t, "select", k[1], indexers={t: "_T"}, drop=True)

    # Share from aviation
    # TODO Compute this using an emission factor and the energy use totals
    k_share = genno.Key("AIR emission share", tuple("eny"))
    c.add(k_share, aviation_share)

    # Product of the total and aviation share → aviation emissions
    c.add(k[3] / t, "mul", k[2] / t, k_share)

    # Re-add the "t" dimension:
    # - +1 for labels with missing data
    # - -1 for labels with existing data from which the aviation total is subtracted
    c.add(
        "broadcast:t:AIR emissions",
        genno.Quantity(
            [1, 1, -1, -1],
            coords={
                "t": [
                    "Aviation",
                    "Aviation|International",
                    "Road Rail and Domestic Shipping",
                    "_T",
                ]
            },
        ),
    )
    c.add(k[5], "mul", k[3] / t, "broadcast:t:AIR emissions")

    # Add to the input data
    c.add(k[6], "add", k[1], k[5])

    # - Collapse to IAMC "VARIABLE" dimension name
    # - Recombine with other data
    # - Write back to the file
    c.add(
        "target",
        finalize,
        k_input,
        k[6],
        "model name",
        "scenario name",
        path_out=path_out,
    )

    c.get("target")
