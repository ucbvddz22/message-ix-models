"""ENGAGE workflow pieces for reuse with :class:`message_ix_models.Workflow`.

These functions emulate the collective behaviour of :class:`.engage.runscript_main`,
:class:`.engage.scenario_runner` and the associated configuration, but are adapted to be
reusable, particularly in the Workflow pattern used in e.g. :mod:`.projects.navigate`.
"""
import logging
from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from iam_units import convert_gwp
from message_ix import Scenario
from message_ix_models import Context, ScenarioInfo
from message_ix_models.util import identify_nodes
from message_ix_models.util.config import ConfigHelper
from message_ix_models.workflow import Workflow

from message_data.tools.utilities import transfer_demands

from .runscript_main import glb_co2_relation as RELATION_GLOBAL_CO2
from .scenario_runner import ScenarioRunner

log = logging.getLogger(__name__)


@dataclass
class PolicyConfig(ConfigHelper):
    """Configuration for the 3-step ENGAGE workflow for climate policy scenarios."""

    #: Label of the climate policy scenario, often related to a global carbon budget in
    #: Gt CO₂ for the 21st century (varies).
    label: Union[int, str]

    #: Actual quantity of the carbon budget to be imposed, or the value "calc", in which
    #: case the value is calculated from :attr:`label` by
    #: :meth:`.ScenarioRunner.calc_budget`.
    budget: Union[int, Literal["calc"]] = "calc"

    #: Scenario name of a reference scenario for copying demands in :func:`step_1`
    #: and/or :func:`step_2`. See also :func:`.transfer_demands`.
    #:
    #: TODO choose a more informative name.
    low_dem_scen: Optional[str] = None

    #: Information on an optional, second scenario from which to copy tax_emission data
    #: in :func:`step_3`.
    tax_emission_scenario: Dict = field(default_factory=dict)

    #: :obj:`True` to call :func:`.reserve_margin.res_marg.main` at each :func:`solve`.
    reserve_margin: bool = True

    #: Which steps of the ENGAGE workflow to run. Empty list = don't run any steps.
    steps: List[int] = field(default_factory=lambda: [1, 2, 3])

    #: Keyword arguments for :meth:`.message_ix.Scenario.solve`. To replicate the
    #: behaviour of the `macro_params` argument to :meth:`.ScenarioRunner.run`, which in
    #: turn sets the `convergence_issues` argument to :meth:`.ScenarioRunner.solve`,
    #: set max_adjustment to 0.1.
    solve: Dict[str, Any] = field(
        default_factory=lambda: dict(model="MESSAGE-MACRO", max_adjustment=0.2)
    )


def calc_hist_cum_CO2(
    context: Context, scen: Scenario, info: ScenarioInfo
) -> Tuple[float, float]:
    """Calculate historic CO2 emissions.

    Adapted from :meth:`.engage.scenario_runner.ScenarioRunner.calc_hist_cum_CO2`, with
    the following differences:

    - Reported emissions are retrieved for a region with the code "GLB region (R##)",
      based on `context`. This allows the method to work with either R11 or R12 models.

    Returns
    -------
    float
        Cumulative emissions [megatonne CO₂].
    float
        Total duration of historical periods [years].
    """
    # TODO carefully check whether self.scen and scen being different scenarios would
    #      impact behaviour

    # Original comment: Filters years between 2010 and first_model_year
    # TODO clarify the difference between the comment and the code
    years = list(filter(lambda y: 2020 <= y < info.y0, info.set["year"]))

    df = scen.timeseries(
        region=f"GLB region ({context.model.regions})", variable="Emissions|CO2"
    )
    df = df[df.year.isin(years)]

    df_emis = df[["year", "value"]].set_index("year")

    df_duration = scen.par("duration_period", filters={"year": years})[
        ["year", "value"]
    ].set_index("year")

    # Manually set 2018 and 2020 multiplication factor
    df_duration.at[2018, "value"] = 0.5
    df_duration.at[2020, "value"] = 3.5
    # manually set 2018 Value
    df_emis.at[2018, "value"] = 41000
    value = (df_emis * df_duration).value.sum()
    hist_years = df_duration.value.sum()
    return (value, hist_years)


def calc_budget(
    context: Context,
    scenario: Scenario,
    bdgt: Union[float, str],
    method: Union[float, Literal["calc"]],
    type_emission="TCE",
) -> None:
    """Calculate **and apply** budget.

    Adapted from :meth:`.engage.scenario_runner.ScenarioRunner.calc_budget`, with the
    following differences:

    - :func:`.calc_hist_cum_CO2` above is called, instead of the method of the same name
      on ScenarioRunner.
    - `scenario` is an argument, not retrieved from the `scen` attribute of
      ScenarioRunner.
    - :mod:`iam_units` is used for unit conversion.

    .. todo:: Fix confusing semantics, here and/or in the original; e.g. use a `value`
       argument and/or config class.

    Parameters
    ----------
    bdgt : float or str
        If `method` is "calc", this must be the budget expressed as total gigatonnes of
        CO₂ for the period 2010–2100. Otherwise, the argument is ignored.
    method : float or str
        Literal "calc" to calculate a constraint budget based on `bdgt`; otherwise,
        budget constraint value expressed as average Mt C-eq / a.
    """
    from message_data.tools.utilities import add_budget

    if method == "calc":
        info = ScenarioInfo(scenario)

        # Target is provided in cumulative Gt 2010-2100
        value = float(bdgt)
        # Convert Gt CO2 to Mt CO2
        value *= 1000

        # NB the original passed ScenarioRunner.base, rather than ScenarioRunner.scen
        # TODO carefully check whether this distinction meant anything
        hist_cum, hist_year = calc_hist_cum_CO2(context, scenario, info)

        value -= hist_cum
        # Conversion to MtC
        value *= convert_gwp(None, (value, "Mt CO2"), "C").magnitude

        # Divide by the number of years covered. "-2" is because in the calculation of
        # the budget we account for 2 more years than the model sees.
        value /= (info.Y[-1] - info.Y[0] + 10) - 2

        # The amount of years over which the budget is distributed should be 82 years
        # (from 2018 to 2100). The number of historic years are subtracted.
        # (do the extra 10 years need to be added)?
        # value /= 82 - hist_year + 10
    else:
        value = method

    add_budget(scenario, value, type_emission=type_emission)


def solve(context: Context, scenario: Scenario, config: PolicyConfig):
    from message_data.scenario_generation.reserve_margin import res_marg

    if config.reserve_margin:
        res_marg.main(scenario)

    var_list = ["I", "C"]
    if config.solve["model"] == "MESSAGE-MACRO":
        var_list.append("GDP")

    scenario.solve(var_list=var_list, **config.solve)

    # Solve was successful; set default version
    scenario.set_as_default()

    return scenario


def step_0(context: Context, scenario: Scenario, **kwargs) -> Scenario:
    """Preparation for the ENGAGE climate policy workflow.

    These operations must occur no matter which combinations of 1 or more of
    :func:`step_1`, :func:`step_2`, and/or :func:`step_3` are to be run on `scenario`.
    """
    from message_data.tools.utilities import (
        add_AFOLU_CO2_accounting,
        add_alternative_TCE_accounting,
        add_CO2_emission_constraint,
        add_FFI_CO2_accounting,
        remove_emission_bounds,
    )

    try:
        scenario.remove_solution()
    except ValueError:
        pass  # Solution did not exist

    remove_emission_bounds(scenario)

    # Identify the node codelist used by `scenario` (in case it is not set on `context`)
    context.model.regions = identify_nodes(scenario)

    kw = dict(relation_name=RELATION_GLOBAL_CO2, reg=f"{context.model.regions}_GLB")

    # “Step1.3 Make changes required to run the ENGAGE setup” (per .runscript_main)
    log.info("Add separate FFI and AFOLU CO2 accounting")
    add_FFI_CO2_accounting(scenario, **kw)
    add_AFOLU_CO2_accounting(scenario, **kw)

    log.info("Add alternative TCE accounting")
    add_alternative_TCE_accounting(scenario)

    add_CO2_emission_constraint(scenario, **kw, constraint_value=0.0, type_rel="lower")

    return scenario


def step_1(context: Context, scenario: Scenario, config: PolicyConfig) -> Scenario:
    """Step 1 of the ENGAGE climate policy workflow.

    If the :attr:`~.PolicyConfig.method` attribute of `policy_config` is "calc", then
    `scenario` must contain time series data for variable="Emissions|CO2" and
    region="GLB region (R##)"; see :func:`calc_hist_cum_CO2`. This is typically provided
    by the legacy reporting. If the attribute has a float value specifying a budget
    directly, this condition does not apply.
    """
    # Calculate **and apply** budget
    calc_budget(
        context,
        scenario,
        bdgt=config.label,
        method=config.budget,
        type_emission="TCE_CO2",
    )

    if config.low_dem_scen:
        # Retrieve certain demands from a different scenario
        source = Scenario(
            scenario.platform, model=scenario.model, scenario=config.low_dem_scen
        )
        transfer_demands(source, scenario)

    return scenario


def step_2(context: Context, scenario: Scenario, config: PolicyConfig) -> Scenario:
    """Step 2 of the ENGAGE climate policy workflow."""
    from message_data.tools.utilities import add_emission_trajectory

    # Retrieve a pandas.DataFrame with the CO2 emissions trajectory
    #
    # NB this method:
    # - does not use any class or context attributes, so it can be called on any
    #   instance of ScenarioRunner.
    #   TODO separate the method from the class as a stand-alone function
    # - does not require legacy reporting output; only the variable "EMISS", i.e.
    #   `scenario` must have solution data.
    sr = ScenarioRunner(context)
    df = sr.retr_CO2_trajectory(scenario)

    try:
        scenario.remove_solution()
    except ValueError:
        pass  # Solution did not exist

    # Add this trajectory as bound_emission values
    add_emission_trajectory(
        scenario,
        data=df,
        type_emission="TCE_CO2",
        unit="Mt C/yr",
        remove_bounds_emission=True,
    )

    with scenario.transact(message="Remove lower bound on global total CO₂ emissions"):
        name = "relation_lower"
        scenario.remove_par(
            name, scenario.par(name, filters={"relation": [RELATION_GLOBAL_CO2]})
        )

    if config.low_dem_scen:
        # Retrieve certain demands from a different scenario
        source = Scenario(
            scenario.platform, model=scenario.model, scenario=config.low_dem_scen
        )
        transfer_demands(source, scenario)

    return scenario


def step_3(context: Context, scenario: Scenario, config: PolicyConfig) -> Scenario:
    """Step 3 of the ENGAGE climate policy workflow."""
    if config.tax_emission_scenario:
        # Retrieve CO2 prices from a different scenario
        source = Scenario(scenario.platform, **config.tax_emission_scenario)
    else:
        # Retrieve CO2 prices from `scenario` itself
        source = scenario

    # Retrieve a data frame with CO₂ prices
    #
    # NB this method:
    # - does not use any class or context attributes, so it can be called on any
    #   instance of ScenarioRunner.
    #   TODO separate the method from the class as a stand-alone function
    # - does not require legacy reporting output; only the variable "PRICE_EMISSION",
    #   i.e. `source` must have solution data.
    sr = ScenarioRunner(context)
    df = sr.retr_CO2_price(source, new_type_emission="TCE_non-CO2")

    del source  # No longer used → free memory

    try:
        scenario.remove_solution()
    except ValueError:
        pass  # Solution did not exist

    with scenario.transact(message="Add price for TCE_non-CO2"):
        scenario.add_par("tax_emission", df)

    return scenario


def add_steps(
    workflow: Workflow, base: str, config: PolicyConfig, name: Optional[str] = None
) -> str:
    """Add steps to `workflow` for running ENGAGE scenarios on `base`.

    Parameters
    ----------
    workflow
    base
       Prior workflow step/scenario to start from.
    config
       Depending on the :attr:`~.PolicyConfig.steps` attribute, from 0 to 3 workflow
       steps are added.
    name : str, optional
       Name template for the added steps.

    Returns
    -------
    str
        name of the last workflow step added, or `base` if none are added.
    """
    workflow.graph["context"].setdefault("run_reporting_only", False)

    # Base name for the added steps
    name_root = f"{name or base} EN"
    # Label for steps
    label = str(config.label).replace(" ", "_")

    # Model and scenario name for the scenario produced by the base step
    info, _ = workflow.guess_target(base, "scenario")
    # Template for new model/scenario name at each step
    target = f"{info['model']}/{info['scenario']}_ENGAGE_{label}_step-{{}}"

    # Model to solve
    solve_model = config.solve["model"]

    # If config.steps is non-empty, insert step_0. Otherwise, leave empty
    steps = copy(config.steps)
    if len(steps):
        steps.insert(0, 0)

    # Iterate over [0, 1, 2, 3] or fewer ENGAGE `steps`
    s = base  # Current step name
    for step in steps:
        # Duplicate `config` and modify for this particular step
        cfg = deepcopy(config)
        if solve_model == "MESSAGE-MACRO" and step > 1:
            # Give scenario name from which to copy "DEMAND" variable data; data is
            # copied to the "demand" parameter
            cfg.low_dem_scen = target.split("/")[-1].format(step - 1)
        if step == 2:
            # Do not solve MESSAGE-MACRO for step 2, even if doing so for steps 1/3
            cfg.solve.update(model="MESSAGE")

        # Add step
        s = workflow.add_step(
            f"{name_root}{step}",
            s,
            # Get a reference to the function step_0() etc.
            action=globals()[f"step_{step}"],
            # Always clone to new URL; if step_0, then shift the model horizon
            clone=dict(keep_solution=True),
            target=target.format(step),  # Target URL
            config=cfg,
        )

        if step > 0:
            # Add a step to solve the scenario (except for after step_0); update the
            # step name for the next loop iteration
            s = workflow.add_step(f"{s} solved", s, solve, config=cfg)

    return s
