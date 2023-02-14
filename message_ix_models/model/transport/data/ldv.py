"""Data for light-duty vehicles (LDVs) for passenger transport."""
import logging
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Sequence, Union

import pandas as pd
from genno import computations
from ixmp.reporting import RENAME_DIMS, Quantity
from message_ix import make_df
from message_ix_models.model import disutility
from message_ix_models.model.structure import get_codes, get_region_codes
from message_ix_models.util import (
    ScenarioInfo,
    adapt_R11_R12,
    adapt_R11_R14,
    broadcast,
    cached,
    check_support,
    convert_units,
    make_io,
    make_matched_dfs,
    merge_data,
    nodes_ex_world,
    private_data_path,
    same_node,
)
from openpyxl import load_workbook
from sdmx.model import Code

from message_data.model.transport.data.emissions import ef_for_input
from message_data.model.transport.utils import input_commodity_level, path_fallback

log = logging.getLogger(__name__)


def get_ldv_data(context) -> Dict[str, pd.DataFrame]:
    """Generate techno-economic data for light-duty-vehicle technologies.

    Responds to the ``["transport config"]["data source"]["LDV"]`` context setting:

    - :obj:`None`: calls :func:`get_dummy`.
    - “US-TIMES MA3T”: calls :func:`get_USTIMES_MA3T`.

    In both cases, :func:`get_constraints` is used to generate constraints.
    """
    source = context.transport.data_source.LDV

    if source == "US-TIMES MA3T":
        return get_USTIMES_MA3T(context)
    elif source is None:
        return get_dummy(context)
    else:
        raise ValueError(f"invalid source for non-LDV data: {source}")


#: Input file containing structured data about LDV technologies.
#:
#: For R11, this data is from the US-TIMES and MA3T models.
FILE = "ldv-cost-efficiency.xlsx"

#: (parameter name, cell range, units) for data to be read from multiple sheets in the
#: :data:`FILE`.
TABLES = {
    "efficiency": (slice("B3", "Q15"), "Gv km / (GW year)"),
    "inv_cost": (slice("B33", "Q45"), "USD / vehicle"),
    "fix_cost": (slice("B62", "Q74"), "USD / vehicle"),
}


@cached
def read_USTIMES_MA3T(nodes: List[str], subdir=None) -> Dict[str, Quantity]:
    """Read the US-TIMES MA3T data from :data:`FILE`.

    No transformation is performed.

    **NB** this function takes only simple arguments (`nodes` and `subdir`) so that
    :func:`.cached` computes the same key every time to avoid the slow step of opening/
    reading the large spreadsheet. :func:`get_USTIMES_MA3T` then conforms the data to
    particular context settings.
    """
    # Open workbook
    path = private_data_path("transport", subdir or "", FILE)
    wb = load_workbook(path, read_only=True, data_only=True)

    # Tables
    data = defaultdict(list)

    # Iterate over regions/nodes
    for node in map(str, nodes):
        # Worksheet for this region
        sheet_node = node.split("_")[-1].lower()
        sheet = wb[f"MESSAGE_LDV_{sheet_node}"]

        # Read tables for efficiency, investment, and fixed O&M cost
        # NB fix_cost varies by distance driven, thus this is the value for average
        #    driving.
        # TODO calculate the values for modest and frequent driving
        for par_name, (cells, _) in TABLES.items():
            df = pd.DataFrame(list(sheet[cells])).applymap(lambda c: c.value)

            # - Make the first row the headers.
            # - Drop extra columns.
            # - Use 'MESSAGE name' as the technology name.
            # - Melt to long format.
            # - Year as integer.
            # - Assign "node" and "unit" columns.
            # - Drop NA values (e.g. ICE_L_ptrp after the first year).
            data[par_name].append(
                df.iloc[1:, :]
                .set_axis(df.loc[0, :], axis=1)
                .drop(["Technology", "Description"], axis=1)
                .rename(columns={"MESSAGE name": "t"})
                .melt(id_vars=["t"], var_name="y")
                .astype({"y": int})
                .assign(n=node)
                .dropna(subset=["value"])
            )

    # Combine data frames, convert to Quantity
    qty = {}
    for par_name, dfs in data.items():
        qty[par_name] = Quantity(
            pd.concat(dfs, ignore_index=True).set_index(["n", "t", "y"]),
            units=TABLES[par_name][1],
            name=par_name,
        )

    return qty


def read_USTIMES_MA3T_2(nodes: Any, subdir=None) -> Dict[str, Quantity]:
    """Same as :func:`read_USTIMES_MA3T`, but from CSV files."""
    result = {}
    for name in "fix_cost", "efficiency", "inv_cost":
        result[name] = computations.load_file(
            path=private_data_path("transport", subdir or "", f"ldv-{name}.csv"),
            dims=RENAME_DIMS,
            name=name,  # FIXME this appears to not pass through ffill()
        ).ffill("y")
        result[name].name = name

    return result


def get_USTIMES_MA3T(context) -> Dict[str, pd.DataFrame]:
    """Prepare LDV data from US-TIMES and MA3T.

    .. todo:: Some calculations are performed in the spreadsheet; transfer to code.
    .. todo:: Values for intermediate time periods e.g. 2025 are forward-filled from
       the next earlier period, e.g. 2020; interpolate instead.

    Returns
    -------
    dict of (str → pd.DataFrame)
        Data for the ``input``, ``output``, ``capacity_factor, ``technical_lifetime``,
        ``inv_cost``, and ``fix_cost`` parameters.
    """
    # Compatibility checks
    check_support(
        context,
        settings=dict(regions=frozenset(["R11", "R12", "R14"])),
        desc="US-TIMES and MA3T data available",
    )

    # Retrieve configuration and ScenarioInfo
    technical_lifetime = context.transport.ldv_lifetime["average"]
    info = context["transport build info"]
    spec = context["transport spec"]

    # Merge with base model commodity information for io_units() below
    # TODO this duplicates code in .ikarus; move to a common location
    all_info = ScenarioInfo()
    all_info.set["commodity"].extend(get_codes("commodity"))
    all_info.update(spec.add)

    if context.model.regions in ("R12", "R14"):
        # Read data using the R11 nodes
        read_nodes: Sequence[Union[str, Code]] = get_region_codes("R11")
    else:
        read_nodes = nodes_ex_world(info.N)

    # Retrieve the input data
    data = read_USTIMES_MA3T_2(read_nodes, subdir="R11")

    # Convert R11 to R12 or R14 data, as necessary
    if context.model.regions == "R12":
        data = adapt_R11_R12(data)
    elif context.model.regions == "R14":
        data = adapt_R11_R14(data)

    # List of years to include
    target_years = list(filter(lambda y: y >= 2010, info.set["year"]))
    for name, qty in data.items():
        # Select only the periods appearing in the target scenario
        tmp = qty.sel(y=target_years)

        # Fill forward over uncovered periods in the model horizon
        # TODO move this operation upstream
        years = tmp.coords["y"].to_index()
        missing = sorted(set(target_years) - set(years))

        log.info(f"{name}: fill from {years[-1]} → {missing}")
        quantities = [tmp] + [
            computations.relabel(tmp.sel(y=years[-1:]), y={years[-1]: m})
            for m in missing
        ]
        data[name] = computations.concat(*quantities)

    # Convert 'efficiency' into 'input' and 'output' parameter data

    # Reciprocal units and value:
    name = "efficiency"
    qty = data.pop(name)
    base = qty.to_series().reset_index()
    src_units = (1.0 / qty.units).units

    common = dict(mode="all", time="year", time_dest="year", time_origin="year")

    i_o = make_io(
        src=(None, None, f"{src_units:~}"),
        dest=(None, "useful", "Gv km"),
        # Reciprocal value, i.e. from  Gv km / GW a → GW a / Gv km
        efficiency=1.0 / base[name],
        on="input",
        node_loc=base["n"],  # Other dimensions
        technology=base["t"],
        year_vtg=base["y"],
        **common,
    )

    # Assign input commodity and level according to the technology
    result = {}
    result["input"] = (
        input_commodity_level(context, i_o["input"], default_level="final")
        .pipe(broadcast, year_act=info.Y)
        .query("year_act >= year_vtg")
        .pipe(same_node)
    )

    # Convert units to the model's preferred input units for each commodity
    @lru_cache
    def _io_units(t, c, l):  # noqa: E741
        return all_info.io_units(t, c, l)

    target_units = (
        result["input"]
        .apply(
            lambda row: _io_units(row["technology"], row["commodity"], row["level"]),
            axis=1,
        )
        .unique()
    )
    assert 1 == len(target_units)

    result["input"]["value"] = convert_units(
        result["input"]["value"], {"value": (1.0, src_units, target_units[0])}
    )

    # Assign output commodity based on the technology name
    result["output"] = (
        i_o["output"]
        .assign(commodity=lambda df: "transport vehicle " + df["technology"])
        .pipe(broadcast, year_act=info.Y)
        .query("year_act >= year_vtg")
        .pipe(same_node)
    )

    # Add capacity factors and technical lifetimes
    result.update(
        make_matched_dfs(
            base=result["output"],
            capacity_factor=1.0,
            technical_lifetime=technical_lifetime,
        )
    )

    # Transform costs
    for name in "fix_cost", "inv_cost":
        base = data[name].to_series().reset_index()
        result[name] = make_df(
            name,
            node_loc=base["n"],
            technology=base["t"],
            year_vtg=base["y"],
            value=base[name],
            unit=f"{data[name].units:~}",
        )
    result["fix_cost"] = (
        result["fix_cost"]
        .pipe(broadcast, year_act=info.Y)
        .query("year_act >= year_vtg")
    )

    # Compute CO₂ emissions factors
    result.update(ef_for_input(context, result["input"], species="CO2"))

    return result


def get_dummy(context) -> Dict[str, pd.DataFrame]:
    """Generate dummy, equal-cost output for each LDV technology."""
    # Information about the target structure
    info = context["transport build info"]

    # List of years to include
    years = list(filter(lambda y: y >= 2010, info.set["year"]))

    # List of LDV technologies
    all_techs = context.transport.set["technology"]["add"]
    ldv_techs = list(map(str, all_techs[all_techs.index("LDV")].child))

    # 'output' parameter values: all 1.0 (ACT units == output units)
    # - Broadcast across nodes.
    # - Broadcast across LDV technologies.
    # - Add commodity ID based on technology ID.
    output = (
        make_df(
            "output",
            value=1.0,
            year_act=years,
            year_vtg=years,
            unit="Gv km",
            level="useful",
            mode="all",
            time="year",
            time_dest="year",
        )
        .pipe(broadcast, node_loc=info.N[1:], technology=ldv_techs)
        .assign(commodity=lambda df: "transport vehicle " + df["technology"])
        .pipe(same_node)
    )

    # Discard rows for the historical LDV technology beyond 2010
    output = output[~output.eval("technology == 'ICE_L_ptrp' and year_vtg > 2010")]

    # Add matching data for 'capacity_factor' and 'var_cost'
    data = make_matched_dfs(output, capacity_factor=1.0, var_cost=1.0)
    data["output"] = output

    return data


def constraint_data(context) -> Dict[str, pd.DataFrame]:
    """Return constraints on light-duty vehicle technology activity and usage.

    Responds to the ``["transport config"]["constraint"]["LDV growth_activity"]``
    context setting, which should give the allowable *annual* increase/decrease in
    activity of each LDV technology.

    For example, a value of 0.01 means the activity may increase (or decrease) by 1%
    from one year to the next. For periods of length >1 year, this value is compounded.
    """
    config = context.transport

    # Information about the target structure
    info = context["transport build info"]
    years = info.Y[1:]

    # Technologies as a hierarchical code list
    codes = config.set["technology"]["add"]
    ldv_codes = codes[codes.index("LDV")].child

    # All technologies in the spec, as strings
    all_techs = list(map(str, context["transport spec"].add.set["technology"]))

    # List of technologies to constrain, including the LDV technologies, plus the
    # corresponding "X usage by CG" pseudo-technologies
    techs: List[Code] = []
    for t in map(str, ldv_codes):
        techs.extend(filter(lambda _t: t in _t, all_techs))  # type: ignore

    # Constraint value
    annual = config.constraint["LDV growth_activity"]

    data = dict()
    for bound, factor in (("lo", -1.0), ("up", 1.0)):
        par = f"growth_activity_{bound}"
        data[par] = make_df(
            par, value=factor * annual, year_act=years, time="year", unit="-"
        ).pipe(broadcast, node_loc=info.N[1:], technology=techs)

    return data


def usage_data(context) -> Mapping[str, pd.DataFrame]:
    """Generate data for LDV usage technologies.

    These technologies convert commodities like "transport ELC_100 vehicle" (i.e.
    vehicle-distance traveled) into "transport pax RUEAM" (i.e. passenger-distance
    traveled). These data incorporate:

    1. Load factor, in the ``output`` efficiency.
    2. Required consumption of a "disutility" commodity, in ``input``.
    """
    # Add disutility data separately
    spec = context["transport spec disutility"]
    info = deepcopy(context["transport build info"])
    info.set["node"] = nodes_ex_world(info.set["node"])

    data = disutility.data_conversion(info, spec)

    # Read load factor data from file
    q = computations.load_file(
        path_fallback(context.model.regions, "load-factor-ldv.csv"),
        dims={"node": "node_loc"},
        name="load factor",
        units="",
    )

    # Fill load factor values in the "value" column for "output"
    @lru_cache(len(q))
    def _value_for(node_loc):
        return q.sel(node_loc=node_loc).item()

    data["output"]["value"] = data["output"]["node_loc"].apply(_value_for)
    # Alternately —performance seems about the same
    # (
    #     output.merge(q.to_series(), left_on=q.dims, right_index=True)
    #     .drop(columns="value")
    #     .rename(columns={"load factor": "value"})
    # )

    merge_data(data, disutility.data_source(info, spec))

    return data
