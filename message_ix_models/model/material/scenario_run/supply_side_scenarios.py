from message_ix_models import ScenarioInfo
import ixmp
import message_ix
import pandas as pd

"""Infrastructure Supply Side Measures"""

def industry_sector_net_zero_targets(scenario):

    # Add iron and steel net zero target

    s_info = ScenarioInfo(scenario)

    scenario.check_out()

    # Remove the technology diffusion constraints

    remove_years = [2035, 2040, 2045, 2050, 2055, 2060, 2070, 2080, 2090, 2100,2110]
    remove_growth_activity_up = scenario.par("growth_activity_up",
    filters={'technology':['dri_gas_steel','dri_h2_steel', 'eaf_steel'],
    'year_act': remove_years})
    remove_initial_activity_up = scenario.par("initial_activity_up",
    filters={'technology':['dri_gas_steel','dri_h2_steel', 'eaf_steel'],
    'year_act': remove_years})

    scenario.remove_par('growth_activity_up', remove_growth_activity_up)
    scenario.remove_par('initial_activity_up', remove_initial_activity_up)

    # To fix: To facilitate gradual phase out, add back the phasing down constraints

    remove_growth_activity_lo = scenario.par("growth_activity_lo",
    filters={'technology':['bof_steel'], 'year_act': remove_years})
    remove_initial_activity_lo = scenario.par("initial_activity_lo",
    filters={'technology':['bof_steel'], 'year_act': remove_years})
    scenario.remove_par('growth_activity_lo', remove_growth_activity_lo)
    scenario.remove_par('initial_activity_lo', remove_initial_activity_lo)

    remove_soft_activity_up = scenario.par("soft_activity_up",
    filters={'technology':['eaf_steel'], 'year_act': remove_years})
    remove_soft_activity_lo = scenario.par("soft_activity_lo",
    filters={'technology':['bof_steel'], 'year_act': remove_years})
    abs_cost_soft_up = scenario.par("abs_cost_activity_soft_up",
    filters={'technology':['eaf_steel', 'bof_steel'], 'year_act': remove_years})
    level_cost_soft_up = scenario.par("level_cost_activity_soft_up",
    filters={'technology':['eaf_steel', 'bof_steel'], 'year_act': remove_years})

    scenario.remove_par('soft_activity_up', remove_soft_activity_up)
    scenario.remove_par('soft_activity_lo', remove_soft_activity_lo)
    scenario.remove_par('abs_cost_activity_soft_up', abs_cost_soft_up)
    scenario.remove_par('level_cost_activity_soft_up', level_cost_soft_up)

    growth_new_capacity_up = scenario.par("growth_new_capacity_up",
    filters={'technology':['dri_gas_ccs_steel', 'bf_ccs_steel'],
    'year_vtg':remove_years})

    scenario.remove_par('growth_new_capacity_up', growth_new_capacity_up)

    initial_new_capacity_up = scenario.par("initial_new_capacity_up",
    filters={'technology':['dri_gas_ccs_steel', 'bf_ccs_steel'],
    'year_vtg': remove_years})

    scenario.remove_par('initial_new_capacity_up', initial_new_capacity_up)

    # Add net-zero relation
    # Note: In updated SSP implementaiton, 'CO2_Emission' does not exist.
    # The negative coefficients should be read from output parameter.

    co2_ind = scenario.par('relation_activity',
    filters = {'relation':'CO2_ind','technology':["DUMMY_coal_supply",
                                                  "DUMMY_gas_supply"]})

    co2_emi = scenario.par('relation_activity',
    filters = {'relation':'CO2_Emission','technology':["dri_gas_ccs_steel",
                                                        "bf_ccs_steel",]})

    rel_new = pd.concat([co2_ind, co2_emi], ignore_index=True)
    rel_new = rel_new[rel_new['year_rel']>=2070]

    rel_new['node_rel'] = 'R12_GLB'
    rel_new['relation'] = 'steel_sector_target'

    scenario.add_set('relation', 'steel_sector_target')

    # Need to add slack values here. Emissions do not go to zero.
    relation_upper_df =  pd.DataFrame({
    "relation": 'steel_sector_target',
    "node_rel": 'R12_GLB',
    "year_rel": [2070, 2080, 2090, 2100],
    "value": [2.7, 2.5, 2.1, 1.8],
    "unit": "???"
    })

    # relation_lower_df =  pd.DataFrame({
    # "relation": 'steel_sector_target',
    # "node_rel": 'R12_GLB',
    # "year_rel": [2070, 2080, 2090, 2100],
    # "value": 0,
    # "unit": "???"
    # })

    scenario.add_par('relation_activity', rel_new)
    scenario.add_par('relation_upper', relation_upper_df)
    # scenario.add_par('relation_lower', relation_lower_df)

    scenario.commit('Steel sector target added.')


def no_clinker_substitution(scenario):

    # Clinker substituion not allowed
    s_info = ScenarioInfo(scenario)

    nodes = s_info.N
    yv_ya = s_info.yv_ya
    year_act=yv_ya.year_act
    nodes.remove("World")
    nodes.remove("R12_GLB")

    scenario.check_out()

    for n in nodes:
        bound_activity = pd.DataFrame({
             "node_loc": n,
             "technology": 'DUMMY_clay_supply_cement',
             "year_act": year_act,
             "mode": 'M1',
             "time": 'year',
             "value": 0,
             "unit": 't'})
        scenario.add_par("bound_activity_lo", bound_activity)
        scenario.add_par("bound_activity_up", bound_activity)

    scenario.commit('Model changes made.')

def no_ccs(scenario):

    # CCS is not allowed across industry
    # Accelerated carbonation not allowed, 'recycling_cement':['M2']

    s_info = ScenarioInfo(scenario)

    nodes = s_info.N
    yv_ya = s_info.yv_ya
    year_act=yv_ya.year_act
    nodes.remove("World")
    nodes.remove("R12_GLB")

    technologies = {'bf_ccs_steel': ['M2'],
                    'dri_gas_ccs_steel': ['M1'],
                    'clinker_wet_ccs_cement': ['M1'],
                    'clinker_dry_ccs_cement': ['M1'],
                    'meth_bio_ccs': ['fuel', 'feedstock'],
                    'meth_coal_ccs': ['fuel', 'feedstock'],
                    'meth_ng_ccs': ['fuel', 'feedstock'],
                    'gas_NH3_ccs': ['M1'],
                    'coal_NH3_ccs': ['M1'],
                    'biomass_NH3_ccs': ['M1'],
                    'fueloil_NH3_ccs': ['M1'],
                    'recycling_cement':['M2']
                    }

    scenario.check_out()

    for key, value in technologies.items():
        for n in nodes:
            for y in year_act:
                bound_activity = pd.DataFrame({
                     "node_loc": n,
                     "technology": key,
                     "year_act": y,
                     "mode": value,
                     "time": 'year',
                     "value": 0,
                     "unit": 't'})
                scenario.add_par("bound_activity_lo", bound_activity)
                scenario.add_par("bound_activity_up", bound_activity)

    scenario.commit('Model changes made.')

def increased_recycling(scenario):

    s_info = ScenarioInfo(scenario)

    nodes = s_info.N
    yv_ya = s_info.yv_ya
    year_act=yv_ya.year_act
    nodes.remove("World")
    nodes.remove("R12_GLB")

    # IRON & STEEL AND ALUMINUM
    # *************************
    # Increase maximum allowed recycling
    relation_recycling = scenario.par("relation_activity",
                        filters = {"relation": ["maximum_recycling_aluminum",
                                                "max_regional_recycling_steel"],
                                    "technology": ["total_EOL_steel",
                                                    "total_EOL_aluminum"]})
    relation_recycling['value'] = -0.98

    # Lower the recycling costs

    recycling_costs_steel = scenario.par("var_cost",
                      filters = {"technology": ["prep_secondary_steel_1"]})
    recycling_costs_alu = scenario.par("var_cost",
                      filters = {"technology": ["prep_secondary_aluminum_1"]})

    recycling_costs_steel_remove = scenario.par("var_cost",
                      filters = {"technology": ["prep_secondary_steel_2",
                                                "prep_secondary_steel_3"]})
    recycling_costs_alu_remove = scenario.par("var_cost",
                      filters = {"technology": ["prep_secondary_aluminum_2",
                                                "prep_secondary_aluminum_3"]})

    recycling_costs_steel_2 = recycling_costs_steel.copy()
    recycling_costs_steel_2["technology"] = "prep_secondary_steel_2"

    recycling_costs_steel_3 = recycling_costs_steel.copy()
    recycling_costs_steel_3["technology"] = "prep_secondary_steel_3"

    recycling_costs_alu_2 = recycling_costs_alu.copy()
    recycling_costs_alu_2["technology"] = "prep_secondary_aluminum_2"

    recycling_costs_alu_3 = recycling_costs_alu.copy()
    recycling_costs_alu_3["technology"] = "prep_secondary_aluminum_3"

    scenario.check_out()

    scenario.remove_par("var_cost", recycling_costs_steel_remove)
    scenario.remove_par("var_cost", recycling_costs_alu_remove)
    scenario.add_par("relation_activity", relation_recycling)
    scenario.add_par("var_cost", recycling_costs_steel_2)
    scenario.add_par("var_cost", recycling_costs_steel_3)
    scenario.add_par("var_cost", recycling_costs_alu_2)
    scenario.add_par("var_cost", recycling_costs_alu_3)

    # CONCRETE
    # *************************
    # Increase maximum allowed recycling

    relation_recycling_concrete = scenario.par("relation_activity",
                        filters = {"relation": ["max_regional_recycling_cement"],
                                    "technology": ['concrete_production_cement']})
    # 70% (normal share of aggregates that go into concrete production) *
    # 60% (share of primary aggregates that can be replaced by secondary) *
    # (20% Other + 15% plain concrete + 12.5% reinforced concrete + 30% mortar)
    relation_recycling_concrete['value'] = -0.3255
    scenario.add_par("relation_activity", relation_recycling_concrete)

    scenario.commit("Increased recycling limits")

def limit_asphalt_recycling(scenario):

    # Increased recycling modes (M4,M5) are not allowed.
    # M4: Bitumen same, 50% aggregates replaces with RAP
    # M5: Bitumen reduced via rejuvenator agents, 90% RAP

    s_info = ScenarioInfo(scenario)

    nodes = s_info.N
    yv_ya = s_info.yv_ya
    year_act=yv_ya.year_act
    nodes.remove("World")
    nodes.remove("R12_GLB")

    scenario.check_out()

    for n in nodes:
        for y in year_act:
            bound_activity = pd.DataFrame({
                 "node_loc": n,
                 "technology": 'asphalt_mixing',
                 "year_act": y,
                 "mode": ['M4', 'M5'],
                 "time": 'year',
                 "value": 0,
                 "unit": 't'})
            scenario.add_par("bound_activity_lo", bound_activity)
            scenario.add_par("bound_activity_up", bound_activity)

    scenario.commit('Model changes made.')

# def keep_fuel_share(scenario):
