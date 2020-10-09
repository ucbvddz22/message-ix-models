from collections import defaultdict
import logging
import message_ix
import ixmp
import numpy as np
import pandas as pd
from message_ix import make_df
from message_ix_models import ScenarioInfo
from message_ix_models.util import broadcast, same_node

from .util import read_config

import re

# datafile = "China_steel_renamed - test.xlsx" # "China_steel_renamed.xlsx" #

log = logging.getLogger(__name__)

def read_data():
    """Read and clean data from :file:`n-fertilizer_techno-economic.xlsx`."""
    # Ensure config is loaded, get the context
    context = read_config()

    # Shorter access to sets configuration
    sets = context["material"]["set"]

    # Read the file
    data = pd.read_excel(
        context.get_path("material", "n-fertilizer_techno-economic.xlsx"),
        sheet_name="Sheet1",
    )

    # Prepare contents for the "parameter" and "technology" columns
    # FIXME put these in the file itself to avoid ambiguity/error

    # "Variable" column contains different values selected to match each of
    # these parameters, per technology
    params = [
        "inv_cost",
        "fix_cost",
        "var_cost",
        "technical_lifetime",
        "input_fuel",
        "input_elec",
        "input_water",
        "output_NH3",
        "output_water",
        "output_heat",
        "emissions",
        "capacity_factor",
    ]

    param_values = []
    tech_values = []
    for t in sets["technology"]["add"]:
        param_values.extend(params)
        tech_values.extend([t.id] * len(params))

    # Clean the data
    data = (
        # Insert "technology" and "parameter" columns
        data.assign(technology=tech_values, parameter=param_values)
        # Drop columns that don't contain useful information
        .drop(["Model", "Scenario", "Region"], axis=1)
        # Set the data frame index for selection
        .set_index(["parameter", "technology"])
    )

    # TODO convert units for some parameters, per LoadParams.py

    return data


# Read in technology-specific parameters from input xlsx
def process_china_data_tec():

    import numpy as np

    # Ensure config is loaded, get the context
    context = read_config()

    # Read the file
    data_steel_china = pd.read_excel(
        context.get_path("material", context.datafile),
        sheet_name="technologies",
    )

    # Clean the data

    data_steel_china = data_steel_china \
        [['Technology', 'Parameter', 'Level',  \
        'Commodity', 'Mode', 'Species', 'Units', 'Value']] \
        .replace(np.nan, '', regex=True)

    # Combine columns and remove ''
    list_series = data_steel_china[['Parameter', 'Commodity', 'Level', 'Mode']] \
        .apply(list, axis=1).apply(lambda x: list(filter(lambda a: a != '', x)))
    list_ef = data_steel_china[['Parameter', 'Species', 'Mode']] \
        .apply(list, axis=1)

    data_steel_china['parameter'] = list_series.str.join('|')
    data_steel_china.loc[data_steel_china['Parameter'] == "emission_factor", \
        'parameter'] = list_ef.str.join('|')

    data_steel_china = data_steel_china.drop(['Parameter', 'Level', 'Commodity', 'Mode'] \
        , axis = 1)
    data_steel_china = data_steel_china.drop( \
        data_steel_china[data_steel_china.Value==''].index)

    data_steel_china.columns = data_steel_china.columns.str.lower()

    # Unit conversion

    # At the moment this is done in the excel file, can be also done here
    # To make sure we use the same units

    return data_steel_china

# Read in relation-specific parameters from input xlsx
def process_china_data_rel():

    import numpy as np

    # Ensure config is loaded, get the context
    context = read_config()

    # Read the file
    data_steel_china = pd.read_excel(
        context.get_path("material", context.datafile),
        sheet_name="relations",
    )

    return data_steel_china

# Read in time-dependent parameters
# Now only used to add fuel cost for bare model
def read_timeseries():

    import numpy as np

    # Ensure config is loaded, get the context
    context = read_config()

    if context.scenario_info['scenario'] == 'NPi400':
        sheet_name="timeseries_NPi400"
    else:
        sheet_name = "timeseries"

    # Read the file
    df = pd.read_excel(
        context.get_path("material", context.datafile), sheet_name)

    import numbers
    # Take only existing years in the data
    datayears = [x for x in list(df) if isinstance(x, numbers.Number)]

    df = pd.melt(df, id_vars=['parameter', 'technology', 'mode', 'units'], \
        value_vars = datayears, \
        var_name ='year')

    return df

def read_data_generic():
    """Read and clean data from :file:`generic_furnace_boiler_techno_economic.xlsx`."""

    # Ensure config is loaded, get the context
    context = read_config()
    # Shorter access to sets configuration
    # sets = context["material"]["generic"]

    # Read the file
    data_generic = pd.read_excel(
        context.get_path("material", "generic_furnace_boiler_techno_economic.xlsx"),
        sheet_name="generic")

    # Clean the data
    # Drop columns that don't contain useful information

    data_generic= data_generic.drop(['Region', 'Source', 'Description'], axis = 1)

    # Unit conversion

    # At the moment this is done in the excel file, can be also done here
    # To make sure we use the same units

    return data_generic

def read_data_aluminum():
    """Read and clean data from :file:`aluminum_techno_economic.xlsx`."""

    # Ensure config is loaded, get the context
    context = read_config()

    # Read the file
    data_aluminum = pd.read_excel(
        context.get_path("material", "aluminum_techno_economic.xlsx"),
        sheet_name="data")
    # Clean the data

    data_aluminum= data_aluminum.drop(['Region', 'Source', 'Description'], axis = 1)

    data_aluminum_hist = pd.read_excel(context.get_path("material", \
    "aluminum_techno_economic.xlsx"),sheet_name="data_historical", \
    usecols = "A:F")

    return data_aluminum,data_aluminum_hist

def gen_data_aluminum(scenario, dry_run=False):
    """Generate data for materials representation of aluminum."""
    # Load configuration

    config = read_config()["material"]["aluminum"]

    # Information about scenario, e.g. node, year
    s_info = ScenarioInfo(scenario)

    # Techno-economic assumptions
    data_aluminum, data_aluminum_hist = read_data_aluminum()

    # List of data frames, to be concatenated together at end
    results = defaultdict(list)

    #allyears = s_info.set['year']

    # FIX: The years do not include 1980
    allyears = np.arange(1980, 2101, 10).tolist()
    print('All years')
    print(allyears)
    modelyears = s_info.Y #s_info.Y is only for modeling years
    print('Model years')
    print(modelyears)
    nodes = s_info.N
    yv_ya = s_info.yv_ya
    fmy = s_info.y0
    print('first model year')
    print(fmy)

    nodes.remove('World')

    # Iterate over technologies

    for t in config["technology"]["add"]:

        # Obtain the active and vintage years
        av = data_aluminum.loc[(data_aluminum["technology"] == t),'availability']\
        .values[0]

        # For the technologies with lifetime

        if "technical_lifetime" in data_aluminum.loc[(data_aluminum["technology"] \
        == t)]["parameter"].values:
            lifetime = data_aluminum.loc[(data_aluminum["technology"] == t) & \
            (data_aluminum["parameter"]== "technical_lifetime"),'value'].values[0]
            years_df = scenario.vintage_and_active_years()
            years_df = years_df.loc[years_df["year_vtg"]>= av]
            years_df_final = pd.DataFrame(columns=["year_vtg","year_act"])

        # For each vintage adjsut the active years according to technical lifetime
        for vtg in years_df["year_vtg"].unique():
            years_df_temp = years_df.loc[years_df["year_vtg"]== vtg]
            years_df_temp = years_df_temp.loc[years_df["year_act"]< vtg + lifetime]
            years_df_final = pd.concat([years_df_temp, years_df_final], ignore_index=True)

        vintage_years, act_years = years_df_final['year_vtg'], years_df_final['year_act']

        params = data_aluminum.loc[(data_aluminum["technology"] == t),\
        "parameter"].values.tolist()

        # Availability of the technology
        av = data_aluminum.loc[(data_aluminum["technology"] == t),
                               'availability'].values[0]
        modelyears = [year for year in modelyears if year >= av]
        yva = yv_ya.loc[yv_ya.year_vtg >= av]

        # Iterate over parameters
        for par in params:
            split = par.split("|")
            param_name = split[0]

            # Obtain the scalar value for the parameter

            val = data_aluminum.loc[((data_aluminum["technology"] == t) \
            & (data_aluminum["parameter"] == par)),'value'].values[0]

            common = dict(
            year_vtg= yv_ya.year_vtg,
            year_act= yv_ya.year_act,
            mode="M1",
            time="year",
            time_origin="year",
            time_dest="year",)

            # For the parameters which inlcudes index names
            if len(split)> 1:

                if (param_name == "input")|(param_name == "output"):

                    # Assign commodity and level names
                    com = split[1]
                    lev = split[2]

                    df = (make_df(param_name, technology=t, commodity=com, \
                    level=lev, value=val, unit='t', **common)\
                    .pipe(broadcast, node_loc=nodes).pipe(same_node))

                    results[param_name].append(df)

                elif param_name == "emission_factor":
                    # Assign the emisson type
                    emi = split[1]

                    df = (make_df(param_name, technology=t,value=val,\
                    emission=emi, unit='t', **common).pipe(broadcast, \
                    node_loc=nodes))
                    results[param_name].append(df)

            # Rest of the parameters except input,output and emission_factor

            else:
                df = (make_df(param_name, technology=t, value=val,unit='t', \
                **common).pipe(broadcast, node_loc=nodes))
                results[param_name].append(df)

    # Add the dummy alluminum demand

    values = gen_mock_demand_aluminum(scenario)

    demand_al = (make_df("demand", commodity= "aluminum", \
    level= "demand_aluminum", year = modelyears, value=values, unit='Mt',\
    time= "year").pipe(broadcast, node=nodes))

    results["demand"].append(demand_al)

    # Add historical data

    for tec in data_aluminum_hist["technology"].unique():

        y_hist = [y for y in allyears if y < fmy]
        common_hist = dict(
            year_vtg= y_hist,
            year_act= y_hist,
            mode="M1",
            time="year",)

        val_act = data_aluminum_hist.\
        loc[(data_aluminum_hist["technology"]== tec), "production"]

        df_hist_act = (make_df("historical_activity", technology=tec, \
        value=val_act, unit='Mt', **common_hist).pipe(broadcast, node_loc=nodes))

        results["historical_activity"].append(df_hist_act)

        c_factor = data_aluminum.loc[((data_aluminum["technology"]== tec) \
                    & (data_aluminum["parameter"]=="capacity_factor")), "value"].values

        val_cap = data_aluminum_hist.loc[(data_aluminum_hist["technology"]== tec), \
                                        "new_production"] / c_factor

        df_hist_cap = (make_df("historical_new_capacity", technology=tec, \
        value=val_cap, unit='Mt', **common_hist).pipe(broadcast, node_loc=nodes))

        results["historical_new_capacity"].append(df_hist_cap)

    results = {par_name: pd.concat(dfs) for par_name, dfs in results.items()}

    add_scrap_prices(scenario)

    return results

#TODO: Add historical data ?
def gen_data_generic(scenario, dry_run=False):
    # Load configuration

    # Load configuration
    config = read_config()["material"]["generic_set"]

    # Information about scenario, e.g. node, year
    s_info = ScenarioInfo(scenario)

    # Techno-economic assumptions
    data_generic = read_data_generic()

    # List of data frames, to be concatenated together at end
    results = defaultdict(list)

    # For each technology there are differnet input and output combinations
    # Iterate over technologies

    allyears = s_info.set['year']
    modelyears = s_info.Y #s_info.Y is only for modeling years
    nodes = s_info.N
    yv_ya = s_info.yv_ya
    fmy = s_info.y0

    # 'World' is included by default when creating a message_ix.Scenario().
    # Need to remove it for the China bare model
    nodes.remove('World')

    for t in config["technology"]["add"]:

        # years = s_info.Y
        params = data_generic.loc[(data_generic["technology"] == t),"parameter"]\
        .values.tolist()

        # Availability year of the technology
        av = data_generic.loc[(data_generic["technology"] == t),'availability'].\
        values[0]
        modelyears = [year for year in modelyears if year >= av]
        yva = yv_ya.loc[yv_ya.year_vtg >= av, ]

        # Iterate over parameters (e.g. input|coal|final|low_temp)
        for par in params:
            split = par.split("|")
            param_name = par.split("|")[0]

            val = data_generic.loc[((data_generic["technology"] == t) & \
            (data_generic["parameter"] == par)),'value'].values[0]

            # Common parameters for all input and output tables

            common = dict(
            year_vtg= yva.year_vtg,
            year_act= yva.year_act,
            time="year",
            time_origin="year",
            time_dest="year",)

            if len(split)> 1:

                if (param_name == "input")|(param_name == "output"):

                    com = split[1]
                    lev = split[2]
                    mod = split[3]

                    # Store the available modes for a technology

                    mode_list.append(mod)

                    df = (make_df(param_name, technology=t, commodity=com, \
                    level=lev, mode=mod, value=val, unit='t', **common).\
                    pipe(broadcast, node_loc=nodes).pipe(same_node))

                    results[param_name].append(df)

                elif param_name == "emission_factor":
                    emi = split[1]
                    mod = data_generic.loc[((data_generic["technology"] == t) \
                    & (data_generic["parameter"] == par)),'value'].values[0]

                    # TODO: Now tentatively fixed to one mode. Have values for the other mode too
                    df = (make_df(param_name, technology=t,value=val,\
                    emission=emi, mode="low_temp", unit='t', **common).pipe(broadcast, \
                    node_loc=nodes))

                    for m in np.unique(np.array(mode_list)):
                        df = (make_df(param_name, technology=t,value=val,\
                        emission=emi,mode= m, unit='t', **common)\
                        .pipe(broadcast, node_loc=nodes))

                        results[param_name].append(df)

            # Rest of the parameters apart from input, output and emission_factor
            else:

                df = (make_df(param_name, technology=t, value=val,unit='t', \
                **common).pipe(broadcast, node_loc=nodes))

                results[param_name].append(df)

    results = {par_name: pd.concat(dfs) for par_name, dfs in results.items()}

    return results

def gen_data_variable(scenario, dry_run=False):

    # Information about scenario, e.g. node, year
    s_info = ScenarioInfo(scenario)

    # Generates variables costs for dummy technologies

    data_vc = read_var_cost()
    tec_vc = set(data_vc.technology)

    # List of data frames, to be concatenated together at end
    results = defaultdict(list)

    allyears = s_info.set['year']
    modelyears = s_info.Y #s_info.Y is only for modeling years
    nodes = s_info.N
    yv_ya = s_info.yv_ya
    print("YVYA")
    print(yv_ya)
    fmy = s_info.y0

    nodes.remove('World')

    #for t in config['technology']['add']:
    for t in tec_vc:
    # Special treatment for time-varying params
    #if t in tec_vc:

        common = dict(
            time="year",
            time_origin="year",
            time_dest="year",)

        param_name = "var_cost"
        val = data_vc.loc[(data_vc["technology"] == t), 'value']
        units = data_vc.loc[(data_vc["technology"] == t),'units'].values[0]
        mod = data_vc.loc[(data_vc["technology"] == t), 'mode']
        yr = data_vc.loc[(data_vc["technology"] == t), 'year']

        df = (make_df(param_name, technology=t, value=val,unit='t', \
             year_vtg=yr, year_act=yr, mode=mod, **common).pipe(broadcast, \
             node_loc=nodes))
        results[param_name].append(df)

    # Concatenate to one data frame per parameter
    results = {par_name: pd.concat(dfs) for par_name, dfs in results.items()}

    return results


def gen_data_steel(scenario, dry_run=False):
    """Generate data for materials representation of steel industry.

    """
    # Load configuration
    config = read_config()["material"]["steel"]

    # Information about scenario, e.g. node, year
    s_info = ScenarioInfo(scenario)

    # Techno-economic assumptions
    data_steel = process_china_data_tec()
    # Special treatment for time-dependent Parameters
    data_steel_vc = read_timeseries()
    tec_vc = set(data_steel_vc.technology) # set of tecs with var_cost

    # List of data frames, to be concatenated together at end
    results = defaultdict(list)

    # For each technology there are differnet input and output combinations
    # Iterate over technologies

    allyears = s_info.set['year'] #s_info.Y is only for modeling years
    modelyears = s_info.Y #s_info.Y is only for modeling years
    nodes = s_info.N
    yv_ya = s_info.yv_ya
    fmy = s_info.y0

    print(allyears, modelyears, fmy)

    nodes.remove('World') # For the bare model

    # for t in s_info.set['technology']:
    for t in config['technology']['add']:

        params = data_steel.loc[(data_steel["technology"] == t),\
            "parameter"].values.tolist()

        # # Special treatment for time-varying params
        # if t in tec_vc:
        #     common = dict(
        #         time="year",
        #         time_origin="year",
        #         time_dest="year",)
        #
        #     param_name = "var_cost"
        #     val = data_steel_vc.loc[(data_steel_vc["technology"] == t), 'value']
        #     units = data_steel_vc.loc[(data_steel_vc["technology"] == t), \
        #     'units'].values[0]
        #     mod = data_steel_vc.loc[(data_steel_vc["technology"] == t), 'mode']
        #     yr = data_steel_vc.loc[(data_steel_vc["technology"] == t), 'year']
        #
        #     df = (make_df(param_name, technology=t, value=val,\
        #     unit='t', year_vtg=yr, year_act=yr, mode=mod, **common).pipe(broadcast, \
        #     node_loc=nodes))
        #
        #     print(param_name, df)
        #     results[param_name].append(df)

            param_name = data_steel_vc.loc[(data_steel_vc["technology"] == t), 'parameter']

            for p in set(param_name):
                val = data_steel_vc.loc[(data_steel_vc["technology"] == t) \
                    & (data_steel_vc["parameter"] == p), 'value']
                units = data_steel_vc.loc[(data_steel_vc["technology"] == t) \
                    & (data_steel_vc["parameter"] == p), 'units'].values[0]
                mod = data_steel_vc.loc[(data_steel_vc["technology"] == t) \
                    & (data_steel_vc["parameter"] == p), 'mode']
                yr = data_steel_vc.loc[(data_steel_vc["technology"] == t) \
                    & (data_steel_vc["parameter"] == p), 'year']

                df = (make_df(p, technology=t, value=val,\
                unit='t', year_vtg=yr, year_act=yr, mode=mod, **common).pipe(broadcast, \
                node_loc=nodes))

                print("time-dependent::", p, df)
                results[p].append(df)

        # Iterate over parameters
        for par in params:

            # Obtain the parameter names, commodity,level,emission
            split = par.split("|")
            param_name = split[0]
            # Obtain the scalar value for the parameter
            val = data_steel.loc[((data_steel["technology"] == t) \
            & (data_steel["parameter"] == par)),'value'].values[0]

            common = dict(
                year_vtg= yv_ya.year_vtg,
                year_act= yv_ya.year_act,
                # mode="M1",
                time="year",
                time_origin="year",
                time_dest="year",)

            # For the parameters which inlcudes index names
            if len(split)> 1:

                print('1.param_name:', param_name, t)
                if (param_name == "input")|(param_name == "output"):

                    # Assign commodity and level names
                    com = split[1]
                    lev = split[2]
                    mod = split[3]

                    df = (make_df(param_name, technology=t, commodity=com, \
                    level=lev, value=val, mode=mod, unit='t', **common)\
                    .pipe(broadcast, node_loc=nodes).pipe(same_node))

                elif param_name == "emission_factor":

                    # Assign the emisson type
                    emi = split[1]
                    mod = split[2]

                    df = (make_df(param_name, technology=t, value=val,\
                    emission=emi, mode=mod, unit='t', **common).pipe(broadcast, \
                    node_loc=nodes))

                else: # time-independent var_cost
                    mod = split[1]
                    df = (make_df(param_name, technology=t, value=val, \
                    mode=mod, unit='t', \
                    **common).pipe(broadcast, node_loc=nodes))

                results[param_name].append(df)

            # Parameters with only parameter name
            else:
                print('2.param_name:', param_name)
                # # Historical years are earlier than firstmodelyear
                # y_hist = [y for y in allyears if y < fmy]
                # # print(y_hist, fmy, years)
                # if re.search("historical_", param_name):
                #     common_hist = dict(
                #         year_vtg= y_hist,
                #         year_act= y_hist,
                #         # mode="M1",
                #         time="year",)
                #
                #     df = (make_df(param_name, technology=t, value=val, unit='t', \
                #     **common_hist).pipe(broadcast, node_loc=nodes))
                #     # print(common_hist, param_name, t, nodes, val, y_hist)
                # else:
                df = (make_df(param_name, technology=t, value=val, unit='t', \
                **common).pipe(broadcast, node_loc=nodes))

                results[param_name].append(df)

    # Create external demand param
    parname = 'demand'
    demand = gen_mock_demand_steel()
    df = (make_df(parname, level='demand', commodity='steel', value=demand, unit='t', \
        year=modelyears, **common).pipe(broadcast, node=nodes))
    results[parname].append(df)

    # Concatenate to one data frame per parameter
    results = {par_name: pd.concat(dfs) for par_name, dfs in results.items()}

    return results



DATA_FUNCTIONS = [
    gen_data_steel,
    gen_data_generic,
    gen_data_aluminum,
    gen_data_variable
]


# Try to handle multiple data input functions from different materials
def add_data(scenario, dry_run=False):
    """Populate `scenario` with MESSAGE-Transport data."""
    # Information about `scenario`
    info = ScenarioInfo(scenario)

    # Check for two "node" values for global data, e.g. in
    # ixmp://ene-ixmp/CD_Links_SSP2_v2.1_clean/baseline
    if {"World", "R11_GLB"} < set(info.set["node"]):
        log.warning("Remove 'R11_GLB' from node list for data generation")
        info.set["node"].remove("R11_GLB")

    for func in DATA_FUNCTIONS:
        # Generate or load the data; add to the Scenario
        log.info(f'from {func.__name__}()')
        add_par_data(scenario, func(scenario), dry_run=dry_run)

    log.info('done')


# Generate a fake steel demand
def gen_mock_demand_steel():
    import numpy as np

    modelyears = s_info.Y
    fmy = s_info.y0

    # True steel use 2010 (China) = 537 Mt/year
    demand2010 = 537
    # https://www.worldsteel.org/en/dam/jcr:0474d208-9108-4927-ace8-4ac5445c5df8/World+Steel+in+Figures+2017.pdf
    gdp_growth = [0.121448215899944, 0.0733079014579874, 0.0348154093342843, \
        0.021827616787921, 0.0134425983942219, 0.0108320197485592, \
        0.00884341208063,0.00829374133206562, 0.00649794573935969, 0.00649794573935969]
    baseyear = list(range(2020, 2110+1, 10))

    gr = [(x+1) for x in gdp_growth]

    demand = np.cumprod(gr) * demand2010
    demand_interp = np.interp(modelyears, baseyear, demand)

    return demand.tolist()
def gen_mock_demand_aluminum(scenario):

    # 17.3 Mt in 2010 to match the historical production from IAI.
    # This is the amount right after electrolysis.

    # The future projection of the demand: Increases by half of the GDP growth rate.
    # Starting from 2020.
    context = read_config()
    s_info = ScenarioInfo(scenario)
    modelyears = s_info.Y #s_info.Y is only for modeling years

    gdp_growth = pd.Series([0.121448215899944, 0.0733079014579874, \
                        0.0348154093342843, 0.021827616787921,\
                        0.0134425983942219, 0.0108320197485592, \
                        0.00884341208063, 0.00829374133206562, \
                        0.00649794573935969],index=pd.Index(modelyears, \
                                                            name='Time'))
    # Values in 2010 from IAI.
    fin_to_useful = 0.971
    useful_to_product = 0.866

    i = 0
    values = []
    val = (17.3 * (1+ 0.147718884937996/2) ** context.time_step)
    values.append(val)

    for element in gdp_growth:
        i = i + 1
        if i < len(modelyears):
            val = (val * (1+ element/2) ** context.time_step)
            values.append(val)

    # Adjust the demand according to old scrap level.

    values = [x * fin_to_useful * useful_to_product for x in values]

    return values

def get_data(scenario, context, **options):
    """Data for the bare RES."""
    if context.res_with_dummies:
        dt = get_dummy_data(scenario)
        print(dt)
        return dt
    else:
        return dict()


def get_dummy_data(scenario, **options):
    """Dummy data for the bare RES.

    Currently this contains:
    - A dummy 1-to-1 technology taking input from (dummy, primary) and output
      to (dummy, useful).
    - A dummy source technology for (dummy, primary).
    - Demand for (dummy, useful).

    This ensures that the model variable ACT has some non-zero entries.
    """
    info = ScenarioInfo(scenario)

    common = dict(
        node=info.N[0],
        node_loc=info.N[0],
        node_origin=info.N[0],
        node_dest=info.N[0],
        technology="dummy",
        year=info.Y,
        year_vtg=info.Y,
        year_act=info.Y,
        mode="all",
        # No subannual detail
        time="year",
        time_origin="year",
        time_dest="year",
    )

    data = make_io(
        src=("dummy", "primary", "GWa"),
        dest=("dummy", "useful", "GWa"),
        efficiency=1.,
        on='input',
        # Other data
        **common
    )

    # Source for dummy
    data["output"] = data["output"].append(
        data["output"].assign(technology="dummy source", level="primary")
    )

    data.update(
        make_matched_dfs(
            data["output"],
            capacity_factor=1.,
            technical_lifetime=10,
            var_cost=1,
        )
    )

    common.update(dict(
        commodity="dummy",
        level="useful",
        value=1.,
        unit="GWa",
    ))
    data["demand"] = make_df("demand", **common)

    return data

# def add_scrap_prices(scenario):
#
#     context = read_config()
#     s_info = ScenarioInfo(scenario)
#     nodes = s_info.N
#     modelyears = s_info.Y
#     nodes.remove('World')
#
#     # Distinguish the share and total technologies
#
#     total = ['prep_secondary_aluminum_1', 'prep_secondary_aluminum_2', \
#      'prep_secondary_aluminum_3']
#
#     #total = ["scrap_recovery_aluminum"]
#
#     # Add technology category for total
#
#     scenario.add_cat('technology', 'type_total', total)
#
#     no = 0
#     for tech in total:
#
#         # Add technology category for shares
#
#         no = no + 1
#         no_shr = str(no)
#         type_tec_shr = 'type' + no_shr
#         share = 'scrap_availability_' + no_shr
#
#         scenario.add_set('shares',share)
#         scenario.add_cat('technology',type_tec_shr, [tech])
#
#         # Map shares
#
#         map_share = pd.DataFrame({'shares': [share],
#                        'node_share': nodes,
#                        'node': nodes,
#                        'type_tec': type_tec_shr,
#                        'mode': 'M1',
#                        'commodity': 'aluminum',
#                        'level': 'old_scrap',})
#         scenario.add_set('map_shares_commodity_share', map_share)
#
#
#         # Map total
#
#         map_total = pd.DataFrame({'shares': [share],
#                    'node_share': nodes,
#                    'node': nodes,
#                    'type_tec': 'type_total',
#                    'mode': 'M1',
#                    'commodity': 'aluminum',
#                    'level': 'old_scrap',})
#         scenario.add_set('map_shares_commodity_total', map_total)
#
#         # Add the upper bound
#
#         up_share = pd.DataFrame({'shares': share,
#        'node_share': nodes[0],
#        'year_act': modelyears,
#        'time': 'year',
#        'value': 0.333,
#        'unit': '%',})
#
#         print(up_share)
#
#         scenario.add_par('share_commodity_up', up_share)

def add_scrap_prices(scenario):

    context = read_config()
    s_info = ScenarioInfo(scenario)
    nodes = s_info.N
    modelyears = s_info.Y
    nodes.remove('World')

    total = ['prep_secondary_aluminum_1', 'prep_secondary_aluminum_2', \
     'prep_secondary_aluminum_3']

    no = 0
    for tech in total:

        # Add technology category for shares

        no = no + 1
        relation = 'scrap_availability_' + str(no)

        scenario.add_set('relation',relation)

        # relation_activity for the technology

        nodes_new = nodes * len(modelyears)

        rel_act = pd.DataFrame({
                        'relation': relation,
                        'node_rel': nodes_new,
                        'year_rel': modelyears,
                        'node_loc': nodes_new,
                        'technology': tech,
                        'year_act': modelyears,
                        'mode': 'M1',
                        "value": 1,
                        'unit': '-',
                        })

        # 1/3 of the old scrap is available. 0.24512 the amount of old scrap.
        # Is there a way to obtain it without hard-coding ?

        # Chnage the availability

        if no == 1:
            val = 1/4
        if no == 2:
            val = 1/4
        if no== 3:
            val = 1/2

        rel_act_rec = pd.DataFrame({
                        'relation': relation,
                        'node_rel': nodes_new,
                        'year_rel': modelyears,
                        'node_loc': nodes_new,
                        'technology': "scrap_recovery_aluminum",
                        'year_act': modelyears,
                        'mode': 'M1',
                        "value": -val * 0.24512,
                        'unit': '-',
                        })

        scenario.add_par("relation_activity", rel_act)
        scenario.add_par("relation_activity", rel_act_rec)

        # Add the upper bound

        upper = pd.DataFrame({'relation': relation,
                               'node_rel': nodes_new,
                               'year_rel': modelyears,
                               'value': 0,
                               'unit': '???',})

        scenario.add_par("relation_upper",upper)
