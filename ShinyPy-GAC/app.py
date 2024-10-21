import io
import re
from zipfile import ZipFile
from ixpy.psdmix import PSDMIX
from shiny import App, reactive, render, ui
import plotly.express as px
import numpy as np
import pandas as pd
from shinywidgets import render_widget, output_widget
from NOMAds.adsa import run_adsa
from NOMAds.trm import run_trm
from NOMAds.iast import predict_iast_multi
from NOMAds.utils import convert_K, mass_to_massC_or_mol
from htmltools import HTML
from molmass import Formula, FormulaError
from pint import UnitRegistry
import PSDM
from PSDM_functions import foul_params
import json
from win32api import GetSystemMetrics
from faicons import icon_svg
from pathlib import Path

################
# Layout
################
WIDTH_UNIT = "150px"
WIDTH_BUTTON = "400px"


# ui.input_action_button = partial(ui.input_action_button, width=WIDTH_BUTTON)
def tooltip(name, tip, id=None):
    return ui.tooltip(ui.span(name, " ", icon_svg("circle-question")), tip, id=id)

if (p:=Path(__file__).parent / "about.md").exists():
    with open(p) as file:
        about_md ="".join(file.readlines())
else:
    about_md=""

msgs = {
    "no_adsprops": "<h4>Scenario: {scenario!r} step {step}:</h4>No <b>Adsorption Properties</b> for Adsorbent {adsorbent_name!r} and Adsorbates {adsorbate_names}. <br>Skipping"
}

units = {
    "concentration": ["mg/L", "ug/L", "mol/L", "mmol/L"],
    "length": ["cm", "mm", "m"],
    "velocity": ["cm/s", "m/s", "m/min", "m/h"],
    "density": ["g/ml", "kg/L"],
    "temperature": ["degreeC"],
    "pressure": ["Pa", "hPa", "kPa", "Torr"],
    "flowrate": ["L/h", "L/min", "m**3/h", "m**3/min"],
    "molar_weight": ["g/mol"],
    "molar_volume": ["mol/L"],
    "diffusivity": ["cm2/s"],
    "K": [
        "(ug/g)/((ug/L)**{n})",
        "(ug/mg)/((ug/L)**{n})",
        "(mg/g)/((mg/L)**{n})",
        "(mmol/mg)/((mmol/L)**{n})",
    ],
    "molar_equivalent": ["meq/L"],
}


def input_mask(definition: dict) -> list:
    """dynamically generate layout from dict-specification

    Args:
        definition (dict): {id: {"tag": tag, "value": value, "unit": unit}, ...}

    Returns:
        list: list of ui elements
    """
    layout = []
    for id, vals in definition.items():
        inp = ui.input_numeric(
            id,
            vals["tag"],
            vals["value"],
            min=vals.get("min", 0),
            max=vals.get("max"),
            step=vals.get("step"),
        )
        if not vals["unit"]:
            layout.append(ui.row(ui.column(6, inp)))
        else:
            unit = ui.input_select(
                id + "_unit", "Unit", choices=units[vals["unit"]], width=WIDTH_UNIT
            )
            layout.append(ui.row(ui.column(6, inp), ui.column(6, unit)))
    return layout


import_panel = ui.nav_panel(
    ui.span("Import / Export ", icon_svg("file-import")),
    ui.input_file(
        "import_data",
        tooltip("Load Input Data", "previously saved .json file(s)"),
        accept=[".json"],
        multiple=True,
    ),
    ui.download_button("save_data", "Save Input Data", width=WIDTH_BUTTON),
)


adsorbent_mask = {
    "ads_rad": {"tag": "Particle Radius", "value": 0, "unit": "length"},
    "ads_epor": {
        "tag": "Bed Porosity",
        "value": 0.5,
        "unit": None,
        "min": 0,
        "max": 1,
        "step": 0.1,
    },
    "ads_rhof": {"tag": "Particle Density", "value": 0, "unit": "density"},
    "ads_rhop": {"tag": "Apparent Density", "value": 0, "unit": "density"},
    "ads_Q": {
        "tag": tooltip("Capacity Q", "only relevant for IX"),
        "value": 0,
        "unit": "molar_equivalent",
    },
}

adsorbent_mask_adv = {
    "ads_psdfr": {
        "tag": tooltip(
            "PSDFR",
            "Surface to Pore Diffusion Flux Ratio. For liquid phase cases where there is little or no background organic matter (such as many industrial by-product streams) and a single or multiple components are present, an SPDFR between 2.0 and 8.0 can be used",
        ),
        "value": 5,
        "unit": None,
    },
    "ads_tortu": {"tag": tooltip("Tortuosity", "TIP"), "value": 1, "unit": None},
    "ads_kf": {
        "tag": tooltip(HTML("k<sub>f</sub>"), "Film Transfer Diffusion Coefficient"),
        "value": 0,
        "unit": "velocity",
        "min": 0,
    },
    "ads_dp": {
        "tag": tooltip(HTML("D<sub>P</sub>"), "Pore Diffusivity"),
        "value": 0,
        "unit": "diffusivity",
        "min": 0,
    },
    "ads_ds": {
        "tag": tooltip(HTML("D<sub>S</sub>"), "Liquid Diffusivity"),
        "value": 0,
        "unit": "diffusivity",
        "min": 0,
    },
}


adsorbent_panel = ui.nav_panel(
    ui.span("Adsorbent Characteristics ", icon_svg("circle")),
    ui.layout_columns(
        ui.card(
            ui.row(
                ui.column(6, ui.input_text("ads_name", "Name", "Adsorbent 0")),
                ui.column(6, ui.input_radio_buttons("ads_type", "Type", ["GAC", "IX"])),
            ),
            *input_mask(adsorbent_mask),
            ui.accordion(
                ui.accordion_panel("Advanced", *input_mask(adsorbent_mask_adv)),
                open=False,
            ),
            ui.row(
                ui.input_action_button(
                    "add_ads", "Add/Update Adsorbent", width=WIDTH_BUTTON
                )
            ),
        ),
        ui.card(
            ui.input_select("sel_ads", "Adsorbent Selection", [], size=5),
            ui.input_action_button("load_ads", "Load Adsorbent"),
            ui.input_action_button("del_ads", "Delete Adsorbent"),
        ),
        col_widths=(4, 5),
    ),
)

column_mask = {
    "col_L": {"tag": "Lenght", "value": 0, "unit": "length"},
    # "col_vel": {"tag": "Velocity", "value": 0, "unit": "velocity"},
    "col_diam": {"tag": "Diameter", "value": 0, "unit": "length"},
    "col_flrt": {"tag": "Flow Rate", "value": 0, "unit": "flowrate"},
    "col_temp": {"tag": "Temperature", "value": 20, "unit": "temperature", "max": 50},
}
column_panel = ui.nav_panel(
    ui.span("Column Specifications ", icon_svg("up-down-left-right")),
    ui.layout_columns(
        ui.card(
            ui.row(ui.input_text("col_name", "Column Name", "Column 0")),
            *input_mask(column_mask),
            ui.row(ui.input_action_button("add_col", "Add/Update Column")),
        ),
        ui.card(
            ui.input_select("sel_col", "Column Selection", [], size=5),
            ui.input_action_button("load_col", "Load Column", width=WIDTH_BUTTON),
            ui.input_action_button("del_col", "Delete Column", width=WIDTH_BUTTON),
        ),
        col_widths=(4, 5),
    ),
)

compounds_mask = {
    "comp_MW": {"tag": "Molar Weight", "value": 0, "unit": "molar_weight"},
    "comp_MolarVol": {
        "tag": tooltip("Molar Volume", "needed for empirical correlation"),
        "value": 0,
        "unit": "molar_volume",
    },
    "comp_BP": {
        "tag": tooltip("Boiling Point", "needed for empirical correlation"),
        "value": 0,
        "unit": "temperature",
    },
    "comp_Density": {
        "tag": tooltip("Density", "needed for empirical correlation"),
        "value": 0,
        "unit": "density",
    },
    "comp_valence": {
        "tag": tooltip("Valence", "only needed for IX"),
        "value": 1,
        "unit": None,
        "max": 3,
        "min": 1,
    },
}

compounds_mask_adv = {
    # "comp_Solubility": {"tag": "Solubility", "value": 0, "unit": "concentration"},
    # "comp_VaporPress": {"tag": "Vapor Pressure", "value": 0, "unit": "pressure"},
}

adsprop_mask = {
    "adsprop_K": {
        "tag": tooltip("K", "Freundlichs K (GAC) or selectivity (IX)"),
        "value": 0,
        "unit": "K",
    },
    "adsprop_n": {
        "tag": tooltip("n", "Freundlichs n (GAC)"),
        "value": 0.5,
        "unit": None,
        "max": 1,
        "step": 0.05,
    },
}

adsprop_mask_adv = compounds_mask_adv = {
    "adsprop_kf": {
        "tag": tooltip(HTML("k<sub>f</sub>"), "Film Transfer Diffusion Coefficient"),
        "value": 0,
        "unit": "velocity",
    },
    "adsprop_dp": {
        "tag": tooltip(HTML("D<sub>P</sub>"), "Pore Diffusivity"),
        "value": 0,
        "unit": "diffusivity",
    },
    "adsprop_ds": {
        "tag": tooltip(HTML("D<sub>S</sub>"), "Liquid Diffusivity"),
        "value": 0,
        "unit": "diffusivity",
    },
}

adsorbate_panel = ui.nav_panel(
    ui.span("Adsorbate Properties ", icon_svg("shapes")),
    ui.layout_columns(
        ui.card(
            ui.row(
                ui.column(6, ui.input_text("comp_name", "Name", "Compound 0")),
                ui.column(6, ui.input_text("comp_formula", "Formula", "")),
            ),
            *input_mask(compounds_mask),
            ui.input_action_button("add_comp", "Add/Update Compound"),
        ),
        ui.card(
            ui.input_select("sel_comp", "Compound Selection", [], size=5),
            ui.input_action_button("load_comp", "Load Compound", width=WIDTH_BUTTON),
            ui.input_action_button("del_comp", "Delete Compound", width=WIDTH_BUTTON),
        ),
    ),
)

adsprop_panel = ui.nav_panel(
    ui.span("Adsorption Properties ", icon_svg("right-left")),
    ui.layout_columns(
        ui.card(
            ui.row(
                ui.column(6, ui.input_select("adsprop_adsorbent", "Adsorbent", [])),
                ui.column(6, ui.input_select("adsprop_adsorbate", "Adsorbate", [])),
            ),
            *input_mask(adsprop_mask),
            ui.accordion(
                ui.accordion_panel("Advanced", *input_mask(adsprop_mask_adv)),
                open=False,
            ),
            ui.input_action_button("add_adsprop", "Add/Update Properties"),
        ),
        ui.card(
            ui.input_select(
                "sel_adsprop", "Adsorption Properties Selection", [], size=5
            ),
            ui.input_action_button(
                "load_adsprop", "Load Adsorption Properties", width=WIDTH_BUTTON
            ),
            ui.input_action_button(
                "del_adsprop",
                "Delete Adsorption Properties",
                width=WIDTH_BUTTON,
            ),
        ),
    ),
)


adsa_panel = ui.accordion_panel(
    "Adsorption Analysis",
    ui.layout_columns(
        ui.card(
            ui.input_file(
                "adsa_file",
                tooltip(
                    "Select File with NOM isotherm",
                    "must have columns 'm' (adsorbent dosage in g/L) and 'c' (residual NOM in mg/L)",
                ),
                accept=[".xlsx"],
                multiple=False,
            ),
            ui.layout_columns(
                ui.input_select("adsa_adsorbent", "Adsorbent", []),
                ui.input_numeric(
                    "adsa_MW", "Input Molecular Weight of fictive components", 1000
                ),
                ui.input_selectize(
                    "adsa_k",
                    "Input K Values",
                    choices=[0, 20, 40, 60],
                    multiple=True,
                    options=(
                        {
                            "placeholder": "Enter Value",
                            "create": True,
                        }
                    ),
                ),
                ui.input_numeric("adsa_n", "Input n", 0.25),
                ui.input_task_button("adsa_run", "Run ADSA"),
                col_widths=(6, 6),
            ),
        ),
        ui.card(
            ui.navset_tab(
                ui.nav_panel("Plot", output_widget("adsa_plot")),
                ui.nav_panel("Input", ui.output_data_frame("adsa_df")),
                ui.nav_panel("Output", ui.output_data_frame("adsa_res")),
            ),
        full_screen=True),
        col_widths=(3, 7),
    ),
)

trm_panel = ui.accordion_panel(
    "Tracer Model",
    ui.layout_columns(
        ui.card(
            ui.input_select("trm_adsorbate", "Adsorbate", []),
            ui.input_file(
                "trm_file",
                tooltip(
                    "Select File with Isotherm in NOM presence",
                    "must have columns 'm' (Adsorbent dosage in g/L) and 'c' (Adsorbate concentration in ug/L)",
                ),
                accept=[".xlsx"],
            ),
            ui.input_task_button("trm_run", "Run TRM"),
            ui.input_action_button(
                "trm_accept",
                tooltip("Adopt Output", "This will overwrite Adsorption Properties"),
            ),
        ),
        ui.card(ui.navset_tab(
                ui.nav_panel("Plot", output_widget("trm_plot")),
                ui.nav_panel("Output", ui.output_data_frame("trm_res_df")),
            ), full_screen=True),
    ),
)

nom_panel = ui.nav_panel(ui.span("NOM ", icon_svg("glass-water")), ui.accordion(adsa_panel, trm_panel))

data_panel = ui.nav_panel(
    ui.span("Data ", icon_svg("table")),
    ui.row(
        ui.column(
            3,
            ui.input_file("data_file", None, accept=".xlsx"),
            ui.input_select("data_time_type", "Time Unit", [], selected=""),
            ui.input_select(
                "data_conc_type", "Concentration Unit", [], selected=""
            ),
        ),
        ui.column(
            3,
            ui.input_action_button("data_reset", "Reset Data"),
            ui.input_select("data_influentID", "Influent Keyword", []),
            ui.input_select("data_effluentID", "Effluent Keyword", []),
        ),
    ),
    ui.accordion(
        ui.accordion_panel(
            "Influent Data",
            ui.output_data_frame("infl_df"),
        ),
        ui.accordion_panel(
            "Effluent Data",
            ui.output_data_frame("effl_df"),
        ),
    ),
    value="data_panel",
)

treatment_train_panel = ui.nav_panel(
    ui.span("Treatment Train ", icon_svg("forward")),
    ui.row(
        ui.column(
            5,
            ui.card(
                ui.row(
                    ui.column(
                        5,
                        ui.input_text("scenario_name", "Scenario Name", "Scenario 0"),
                        ui.input_selectize(
                            "scenario_adsorbates", "Adsorbates", [], multiple=True
                        ),
                    ),
                    ui.column(
                        5,
                        ui.input_select(
                            "scenario_water_type",
                            tooltip(
                                "Water Type",
                                "empirical correlation to adjust tortuosity over time",
                            ),
                            list(foul_params["water"].keys()),
                        ),
                        ui.input_select(
                            "scenario_chem_type",
                            tooltip(
                                "Chemical Type",
                                "empirical correlation to adjust Freundlich's K over time",
                            ),
                            list(foul_params["chemical"].keys()),
                            selected="PFAS",
                        ),
                    ),
                ),
                ui.input_action_button("add_scenario", "Add/Update Scenario"),
            ),
        ),
        ui.column(
            5,
            ui.card(
                ui.row(
                    ui.column(
                        5,
                        ui.input_select("sel_scenario", "Select Scenario", [], size=5),
                    )
                ),
                ui.row(
                    ui.column(
                        3, ui.input_action_button("load_scenario", "Load Scenario")
                    ),
                    ui.column(
                        3, ui.input_action_button("del_scenario", "Delete Scenario")
                    ),
                ),
            ),
        ),
    ),
    ui.input_action_button("add_step", "Add step"),
    ui.input_action_button("del_steps", "Delete steps"),
    ui.layout_columns(
        ui.row(
            id="scenario_steps",
        )
    ),
    value="scenario_panel",
)

input_tab = ui.navset_tab(
    import_panel,
    adsorbent_panel,
    column_panel,
    adsorbate_panel,
    adsprop_panel,
    nom_panel,
    data_panel,
    treatment_train_panel,
    id="input_panels",
)


simul_tab = ui.navset_tab(
    ui.nav_panel(
        ui.span("Setup ", icon_svg("sliders")),
        ui.card(
            ui.input_select(
                "sel_run_scenario", "Select Scenarios to Simulate / Save Results from", [], multiple=True
            )
        ),
        ui.card(
            ui.input_slider("sim_nr", "Radial Collocation Points", 3, 18, 7),
            ui.input_slider("sim_nz", "Axial Collocation Points", 3, 18, 12),
            ui.input_slider("sim_ne", "Number of finite Elements", 1, 100, 1),
            ui.input_task_button("run", "Run Analyses"),
            ui.download_button("sim_save", "Save Results"),
        ),
    ),
    ui.nav_panel(
        ui.span("Results ", icon_svg("chart-line")),
        ui.card(
            ui.card_header(
                ui.popover(
                    icon_svg("gear"),
                    ui.input_radio_buttons(
                        "simplot_x_axis",
                        "x-axis",
                        ["time", "BV"],
                        inline=True,
                    ),
                    ui.input_checkbox("simplot_marker", "Draw Marker", True),
                    title="Adjust Plot",
                    placement="top",
                ),
                class_="d-flex justify-content-between align-items-center",
            ),
            output_widget("plot_results"),
            full_screen=True,
        ),
    ),
)

app_ui = ui.page_navbar(
    ui.nav_panel(ui.span("About ", icon_svg("circle-info")),
                 ui.markdown(about_md)), 
    ui.nav_panel(
        ui.span("Input Data ", icon_svg("gears")),
        ui.input_text("project_name", "Project Name", "Project 0"),
        input_tab,
        value="input_tab",
    ),
    ui.nav_panel(ui.span("Simulation ", icon_svg("spinner")), simul_tab, value="sim_tab"),
    
    title="Test",
    fillable=True,
    id="page",
)


def server(input, output, session):
    simulations = {}

    adsorbents = reactive.value({})

    columns = reactive.value({})

    compounds = reactive.value({})

    adsprops = reactive.value({})

    rawdata = {}

    data = reactive.value(
        pd.DataFrame(columns=["type", "time", "concentration", "compound"])
    )

    sheet = reactive.value()

    steps = reactive.value(0)

    scenarios = reactive.value({})

    current_sim = reactive.value()
    # @render.data_frame
    # @reactive.event(input.add_comp)
    # def df_comp_prop():
    #     return render.DataGrid(compounds.reset_index(names=["Property"]), editable=True)

    # @render.data_frame
    # def ads_prop_df():
    #     return render.DataGrid(compounds)

    # @df_comp_prop.set_patch_fn
    # def _(*, patch: render.CellPatch) -> render.CellValue:
    #     if patch["column_index"] == 0:
    #         return compounds.index.values[patch["row_index"]]
    #     return patch["value"]

    def validate_values(values, mask):
        msgs = []
        for key, value in mask.items():
            # valence can be zero
            if key == "comp_valence":
                if not isinstance(values[key], int):
                    msgs.append(f"{value['tag']} must be an integer value")
                continue

            if (values[key] == 0) and (mask[key].get("min") != 0):
                # Q can't be zero for IX
                if key == "ads_Q":
                    if input.ads_type() == "IX":
                        msgs.append(
                            f"{value['tag']} can't be 0 {values[f'{key}_unit']}"
                        )
                else:
                    msgs.append(
                        f"{value['tag']} must be greater than 0 {values[f'{key}_unit']}"
                    )

        if msgs:
            m = ui.modal(HTML("<h1>Invalid Inputs:</h1><br>" + ",<br>".join(msgs)))
            ui.modal_show(m)
        else:
            return True

    ################
    # Load/ Save data
    ################
    @render.download(filename=lambda: f"{input.project_name()}.json")
    def save_data():
        output = {
            "adsorbents": adsorbents(),
            "columns": columns(),
            "compounds": compounds(),
            "adsprops": adsprops(),
            "scenarios": scenarios(),
        }
        # with open("output.json", "w") as f:
        #     json.dump(output, f)

        with io.StringIO() as buf:
            json.dump(output, buf, indent="\t", ensure_ascii=True)
            yield buf.getvalue()

    @reactive.effect
    @reactive.event(input.import_data)
    def _():
        files = input.import_data()

        if not files:
            return

        with ui.Progress(max=len(files)) as p:
            for i, file in enumerate(files):
                name, path = file["name"], file["datapath"]
                p.set(i, message=f"Inporting {name!r}")
                with open(path) as f:
                    upload = json.load(f)

                adsorbents.set(adsorbents() | upload["adsorbents"])
                columns.set(columns() | upload["columns"])
                compounds.set(compounds() | upload["compounds"])
                adsprops.set(adsprops() | upload["adsprops"])
                scenarios.set(scenarios() | upload["scenarios"])

        update_ads()
        update_adsprop()
        update_col()
        update_comp()
        update_scenarios()

        ui.update_text("project_name", value=name.removesuffix(".json"))

    ################
    # ADSA
    ################
    @reactive.calc
    def adsa_run():
        data = adsa_file().copy()
        if data.empty:
            return pd.DataFrame()
        if not input.adsa_adsorbent():
            ui.notification_show(
                "Please select Adsorbent!", type="error", duration=None
            )
            return pd.DataFrame()
        elif not input.adsa_k():
            ui.notification_show(
                "Please specify K values!", type="error", duration=None
            )
            return pd.DataFrame()

        cT0 = data.c[data.m == 0].values
        cTs = data.c[data.m > 0].values
        ms = data.m[data.m > 0].values
        qMs = (cT0 - cTs) / ms
        name = "ADSA"
        Ks = [float(k) for k in input.adsa_k()]
        n = input.adsa_n()
        return run_adsa(cT0, cTs, ms, qMs, name, Ks=Ks, n=n, plot=False)

    @reactive.calc
    def adsa_file():
        file = input.adsa_file()
        if file is None:
            return pd.DataFrame()
        df = pd.read_excel(  # pyright: ignore[reportUnknownMemberType]
            file[0]["datapath"]
        )
        required_cols = ["m", "c"]
        if df.columns.intersection(required_cols).size < 2:
            return pd.DataFrame()

        df = df[required_cols]
        return df

    @render.data_frame
    def adsa_df():
        return render.DataTable(adsa_file())

    @render.data_frame
    @reactive.event(input.adsa_run)
    def adsa_res():
        adsa = adsa_run()
        return render.DataTable(adsa.round(2))

    @render_widget
    @reactive.event(input.adsa_run)
    def adsa_plot():
        adsa_long = (
            adsa_run()
            .rename(
                {
                    "doc": "c_sample",
                    "cges": "c_fitted",
                    "qM": "q_sample",
                    "qR": "q_fitted",
                },
                axis=1,
            )
            .reset_index()
        )
        adsa_long = pd.wide_to_long(
            adsa_long, stubnames=["c", "q"], i="index", j="type", sep="_", suffix=r"\w+"
        )
        scatterplot = px.scatter(adsa_long.reset_index(), x="c", y="q", color="type")
        return scatterplot

    ################
    # TRM
    ################
    @reactive.calc
    def trm_file():
        file = input.trm_file()
        if file is None:
            return pd.DataFrame()
        df = pd.read_excel(  # pyright: ignore[reportUnknownMemberType]
            file[0]["datapath"]
        )
        required_cols = ["m", "c"]
        if df.columns.intersection(required_cols).size < 2:
            m = ui.modal(f"Input data must contain columns 'm' and 'c'!")
            ui.modal_show(m)
            return pd.DataFrame()

        df = df[required_cols]
        return df

    @reactive.calc
    def trm_run():
        data = trm_file().copy()
        if data.empty:
            return
        
        # convert MP concentration to mgC/L
        data.c = data.c * mass_to_massC_or_mol("ug/L", "mgC/L", formula = compounds().get(input.trm_adsorbate())["comp_formula"])
        c0MP = data.c[data.m == 0].values
        cMPs = data.c[data.m > 0].values
        ms = data.m[data.m > 0].values

        KsDOC = [float(k) for k in input.adsa_k()]
        nsDOC = input.adsa_n() * np.ones_like(KsDOC)
        c0sDOC = adsa_run().iloc[0, 3 : 3 + len(KsDOC)].values
        return run_trm(cMPs, ms, c0MP, c0sDOC, KsDOC, nsDOC)

    @reactive.calc
    def trm_res():
        # from input file
        df = trm_file()
        c0MP = df.c[df.m == 0].values
        ms = df.m[df.m > 0].values

        # from ADSA
        KsDOC = [float(k) for k in input.adsa_k()]
        nsDOC = input.adsa_n() * np.ones_like(KsDOC)
        c0sDOC = adsa_run().iloc[0, 3 : 3 + len(KsDOC)].values

        # from TRM
        KMP, nMP = trm_run()

        # combine and predict
        c0R = np.concatenate((c0MP, c0sDOC))
        Ks = np.concatenate(([KMP], KsDOC))
        ns = np.concatenate(([nMP], nsDOC))
        Ls = np.ones_like(ms)
        print(c0R, ns, Ks, ms, Ls)
        return predict_iast_multi(c0R, ns, Ks, ms, Ls)

    @render_widget
    @reactive.event(input.trm_run)
    def trm_plot():
        df = trm_file()
        c0MP = df.c[df.m == 0].values
        ms = df.m[df.m > 0].values
        cMPs = df.c[df.m > 0].values
        qMPs = (c0MP - cMPs) / ms
        real_data = pd.DataFrame(columns=["kind", "c0", "q0"])
        real_data["q0"] = qMPs
        real_data["c0"] = cMPs
        real_data["kind"] = "real"

        trm_data = trm_res()
        trm_data["kind"] = "TRM"
        # trm_data["id"] = trm_data.index
        # trm_data = pd.wide_to_long(trm_data, ["c", "q", "z"], i="id", j="kind").reset_index()

        plot_data = pd.concat([real_data, trm_data])
        scatterplot = px.scatter(plot_data, x="c0", y="q0", color="kind")
        return scatterplot

    @render.data_frame
    @reactive.event(input.trm_run)
    def trm_res_df():
        trm = trm_res()
        return render.DataTable(trm.round(2))
    
    @reactive.effect
    @reactive.event(input.trm_accept)
    def _():
        if ((not input.adsa_adsorbent()) | (not input.trm_adsorbate()) | (not input.adsa_k()) | (trm_file().empty)):
            ui.notification_show("Incomplete Input!", type="error", duration=None)
            return
        ureg = UnitRegistry()
        tracer_comps = {}
        tracer_adsprops = {}
        adsorbent = input.adsa_adsorbent()
        n = input.adsa_n()
        adsorbate = input.trm_adsorbate()
        Ks = input.adsa_k()
        tracer_mask = "TRM_{i}"
        for i, K in enumerate(Ks):
            tracer = tracer_mask.format(i=i)
            tracer_comp = {
                "comp_MW_unit": "g/mol",
                "comp_MolarVol_unit": "mol/L",
                "comp_BP_unit": "degreeC",
                "comp_Density_unit": "g/ml",
                "comp_MW": input.adsa_MW(),
                "comp_MolarVol": 0,
                "comp_BP": 0,
                "comp_Density": 0,
                "comp_valence": 1,
                "comp_name": tracer,
                "comp_formula": "",
            }
            tracer_comps.update({tracer: tracer_comp})
            tracer_adsprop = {
                "adsprop_adsorbent": adsorbent,
                "adsprop_adsorbate": tracer,
                "adsprop_K_unit": units["K"][0],
                "adsprop_kf_unit": "cm/s",
                "adsprop_dp_unit": "cm2/s",
                "adsprop_ds_unit": "cm2/s",
                "adsprop_K": convert_K(float(K), n, "(mg/g)/((mg/L)**{n})", "(ug/g)/((ug/L)**{n})"),
                "adsprop_n": n,
                "adsprop_kf": 0,
                "adsprop_dp": 0,
                "adsprop_ds": 0,
            }
            tracer_adsprops.update({f"{adsorbent}::{tracer}": tracer_adsprop})

        KMP, nMP = trm_run()
        comp_adsprop = {"adsprop_K": KMP, "adsprop_n": nMP}
        adsprops_new = adsprops()
        adsprops_new[f"{adsorbent}::{adsorbate}"].update(comp_adsprop)
        compounds.set(compounds() | tracer_comps)
        adsprops.set(adsprops_new | tracer_adsprops)

        # add c0 to data()
        c0sDOC = adsa_run().iloc[0, 3 : 3 + len(Ks)].values
        tracer_df = pd.DataFrame(columns=["time", "concentration", "compound", "type"])
        if not input.data_conc_type():
            ui.update_select("data_conc_type", choices=["ug/L"], selected="ug/L")

        tracer_df.concentration = (
            c0sDOC * ureg("mg/L").to(input.data_conc_type() if input.data_conc_type() else "ug/L").magnitude
        )
        tracer_df.compound = [tracer_mask.format(i=i) for i in range(len(Ks))]
        tracer_df.time = 0
        tracer_df["type"] = (
            input.data_influentID() if input.data_influentID() else "influent"
        )
        data.set(pd.concat([data(), tracer_df]))
        ui.update_select("data_influentID", choices=data()["type"].unique().tolist())
        update_adsprop()
        update_comp()

        ui.notification_show(
            HTML(
                "Successfully updated <b>Adsorbate Properties</b>,<br><b>Adsorption Properties</b> and <b>Data</b>!"
            )
        )

    ################
    # Adsorbents
    ################

    @reactive.effect
    @reactive.event(input.add_ads)
    def _():
        name = input.ads_name()
        adsorbent = {
            key: input.__dict__["_map"][key]()
            for key in input.__dict__["_map"].keys()
            if key.startswith("ads_")
        }
        if not validate_values(adsorbent, adsorbent_mask | adsorbent_mask_adv):
            return
        adsorbents.set({**adsorbents(), name: adsorbent})
        update_ads()

    @reactive.effect
    @reactive.event(input.load_ads)
    def _():
        name = input.sel_ads()
        if not name:
            return
        for key, val in adsorbents()[name].items():
            if key.endswith("_unit"):
                ui.update_select(key, selected=val)
            elif key != "ads_name":
                ui.update_numeric(key, value=val)
            else:
                ui.update_text(key, value=val)

    @reactive.effect
    @reactive.event(input.del_ads)
    def _():
        name = input.sel_ads()
        if name:
            vals = adsorbents()
            vals.pop(name)
            adsorbents.set(vals)
            update_ads()

    def update_ads():
        choices = list(adsorbents().keys())
        ui.update_select("sel_ads", choices=choices)
        ui.update_select("adsprop_adsorbent", choices=choices)
        ui.update_select("adsa_adsorbent", choices=choices)

    @reactive.effect
    # @reactive.event(input.ads_rhof)
    def _():
        if not (input.ads_rhop() and input.ads_rhof()):
            return

        if input.ads_rhop() > input.ads_rhof():
            m = ui.modal(
                HTML(
                    "<b>Apparent Density</b> must be smaller than <b>Particle Density</b>!"
                )
            )
            ui.modal_show(m)
            ui.update_numeric("ads_rhop", value=input.ads_rhof())

    ################
    # Columns
    ################
    @reactive.effect
    @reactive.event(input.add_col)
    def _():
        name = input.col_name()
        column = {
            key: input.__dict__["_map"][key]()
            for key in input.__dict__["_map"].keys()
            if key.startswith("col_")
        }
        if not validate_values(column, column_mask):
            return
        columns.set({**columns(), name: column})
        update_col()

    @reactive.effect
    @reactive.event(input.load_col)
    def _():
        name = input.sel_col()
        if not name:
            return
        for key, val in columns()[name].items():
            if key.endswith("_unit"):
                ui.update_select(key, selected=val)
            elif key != "col_name":
                ui.update_numeric(key, value=val)
            else:
                ui.update_text(key, value=val)

    @reactive.effect
    @reactive.event(input.del_col)
    def _():
        name = input.sel_col()
        if name:
            vals = columns()
            vals.pop(name)
            columns.set(vals)
            update_col()

    def update_col():
        ui.update_select("sel_col", choices=list(columns().keys()))

    ################
    # Compounds
    ################
    @reactive.effect
    @reactive.event(input.add_comp)
    def _():
        name = input.comp_name()
        compound = {
            key: input.__dict__["_map"][key]()
            for key in input.__dict__["_map"].keys()
            if key.startswith("comp_")
        }
        if not validate_values(compound, compounds_mask):
            return
        compounds.set({**compounds(), name: compound})
        update_comp()

    @reactive.effect
    @reactive.event(input.load_comp)
    def _():
        name = input.sel_comp()
        if not name:
            return
        for key, val in compounds()[name].items():
            if key.endswith("_unit"):
                ui.update_select(key, selected=val)
            elif key != "comp_name":
                ui.update_numeric(key, value=val)
            else:
                ui.update_text(key, value=val)

    @reactive.effect
    @reactive.event(input.del_comp)
    def _():
        name = input.sel_comp()
        if name:
            vals = compounds()
            vals.pop(name)
            compounds.set(vals)
            update_comp()

    def update_comp():
        choices = list(compounds().keys())
        ui.update_select("sel_comp", choices=choices)
        ui.update_select("adsprop_adsorbate", choices=choices)
        ui.update_select("trm_adsorbate", choices=choices)
        ui.update_selectize("scenario_adsorbates", choices=choices)

    @reactive.effect
    @reactive.event(input.comp_formula)
    def _():
        if input.comp_formula():
            try:
                formula = Formula(input.comp_formula())
                ui.update_numeric("comp_MW", value=formula.mass)
            except FormulaError:
                # m = ui.modal(
                #     f"{input.comp_formula()!r} is not a valid formula!",
                #     title="Input Error",
                #     easy_close=True,
                #     footer=None,
                # )
                # ui.modal_show(m)
                ui.update_text("comp_formula", value=input.comp_formula()[:-1])

    ################
    # Adsorption Properties
    ################
    @reactive.effect
    @reactive.event(input.add_adsprop)
    def _():
        if len(adsorbents()) == 0 | len(compounds()) == 0:
            m = ui.modal(
                "Add <b>Adsorbent Characteristic</b> or <b>Adsorbate Property</b> first!"
            )
            ui.modal_show(m)
            return

        name = input.adsprop_adsorbent() + "::" + input.adsprop_adsorbate()
        adsprop = {
            key: input.__dict__["_map"][key]()
            for key in input.__dict__["_map"].keys()
            if key.startswith("adsprop_")
        }
        if not validate_values(adsprop, adsprop_mask):
            return
        adsprops.set({**adsprops(), name: adsprop})
        update_adsprop()

    @reactive.effect
    @reactive.event(input.load_adsprop)
    def _():
        name = input.sel_adsprop()
        if not name:
            return
        for key, val in adsprops()[name].items():
            if key.endswith("_unit"):
                ui.update_select(key, selected=val)
            elif key.removeprefix("adsprop_") not in ["name", "adsorbent", "adsorbate"]:
                ui.update_numeric(key, value=val)
            else:
                ui.update_text(key, value=val)

    @reactive.effect
    @reactive.event(input.del_adsprop)
    def _():
        name = input.sel_adsprop()
        if name:
            vals = adsprops()
            vals.pop(name)
            adsprops.set(vals)
            update_adsprop()

    def update_adsprop():
        ui.update_select("sel_adsprop", choices=list(adsprops().keys()))
        # ui.update_select("adsprop_adsorbate", choices=list(adsprops().keys()))
        # ui.update_selectize("scenario_adsorbates", choices=list(adsprops().keys()))

    ################
    # Treatment Train
    ################
    @reactive.effect
    @reactive.event(input.add_step)
    def _():
        if len(adsorbents()) == 0 | len(columns()) == 0:
            m = ui.modal(
                HTML(
                    "Add <b>Adsorbent Characteristic</b> or <b>Column Specification</b> first!"
                )
            )
            ui.modal_show(m)
            return

        ui.insert_ui(
            ui.column(
                3,
                ui.card(
                    ui.card_header(ui.h2(f"Step {steps()}")),
                    ui.input_select(
                        f"step_adsorbent_{steps()}",
                        "Adsorbent",
                        list(adsorbents().keys()),
                    ),
                    ui.input_select(
                        f"step_column_{steps()}",
                        "Column",
                        list(columns().keys()),
                    ),
                    id=f"step_{steps()}",
                ),
            ),
            selector="#scenario_steps",
            where="beforeEnd",
        )
        steps.set(steps() + 1)

    @reactive.effect(priority=1)
    @reactive.event(input.del_steps, input.load_scenario)
    def _():
        ui.remove_ui(f"#scenario_steps")
        ui.insert_ui(
            ui.layout_columns(ui.row(id="scenario_steps")),
            selector="#del_steps",
            where="afterEnd",
        )
        steps.set(0)

    @reactive.effect
    @reactive.event(input.add_scenario)
    def _():
        if not input.scenario_adsorbates():
            m = ui.modal("Select adsorbates first!")
            ui.modal_show(m)
            return
        if steps() == 0:
            m = ui.modal("Scenario must have at least one step!")
            ui.modal_show(m)
            return

        name = input.scenario_name()

        scenario = {
            "steps": {
                str(i): {
                    "step_adsorbent": input.__dict__["_map"][f"step_adsorbent_{i}"](),
                    "step_column": input.__dict__["_map"][f"step_column_{i}"](),
                }
                for i in range(steps())
            },
            "adsorbates": input.scenario_adsorbates(),
            "water_type": input.scenario_water_type(),
            "chem_type": input.scenario_chem_type(),
        }

        scenarios.set({**scenarios(), name: scenario})
        update_scenarios()

    @reactive.effect
    @reactive.event(input.load_scenario)
    def _():
        name = input.sel_scenario()

        if not name:
            return
        scenario = scenarios()[name]
        for step, values in scenario["steps"].items():
            ui.insert_ui(
                ui.column(
                    3,
                    ui.card(
                        ui.card_header(ui.h2(f"Step {step}")),
                        ui.input_select(
                            f"step_adsorbent_{step}",
                            "Adsorbent",
                            list(adsorbents().keys()),
                            selected=values["step_adsorbent"],
                        ),
                        ui.input_select(
                            f"step_column_{step}",
                            "Column",
                            list(columns().keys()),
                            selected=values["step_column"],
                        ),
                        id=f"step_{step}",
                    ),
                ),
                selector="#scenario_steps",
                where="beforeEnd",
            )
            steps.set(steps() + 1)

        scenario_adsorbates = list(scenario["adsorbates"])
        ui.update_selectize(
            "scenario_adsorbates",
            choices=list(compounds().keys()),
            selected=scenario_adsorbates,
        )
        ui.update_text("scenario_name", value=name)
        ui.update_select("scenario_water_type", selected=scenario["water_type"])
        ui.update_select("scenario_chem_type", selected=scenario["chem_type"])

    @reactive.effect
    @reactive.event(input.del_scenario)
    def _():
        name = input.sel_scenario()
        if name:
            vals = scenarios()
            vals.pop(name)
            scenarios.set(vals)
            update_scenarios()

    def update_scenarios():
        choices = list(scenarios().keys())
        ui.update_select("sel_run_scenario", choices=choices)
        ui.update_selectize("sel_scenario", choices=choices)

    ################
    # Data
    ################
    @reactive.effect
    def data_file():
        nonlocal rawdata
        file = input.data_file()
        if file is not None:
            rawdata = pd.read_excel(  # pyright: ignore[reportUnknownMemberType]
                file[0]["datapath"], sheet_name=None
            )
            if len(rawdata.keys()) == 1:
                sheet.set(list(rawdata.keys())[0])
                return

            m = ui.modal(
                ui.input_select(
                    "data_file_sheet", "Select Sheet", list(rawdata.keys())
                ),
                ui.input_action_button("data_file_sheet_ok", "Accept"),
                footer=None,
            )
            ui.modal_show(m)

    @reactive.effect
    @reactive.event(input.data_file_sheet_ok)
    def _():
        ui.modal_remove()
        sheet.set(input.data_file_sheet())

    @reactive.effect
    @reactive.event(sheet)
    def _():
        required_cols = ["type", "time", "concentration", "compound"]

        if set(required_cols).issubset(rawdata[sheet()].columns):
            new_data = rawdata[sheet()][required_cols]
            new_data["step"] = rawdata[sheet()].get("step", 0)
            new_data["scenario"] = rawdata[sheet()].get("scenario", pd.NA)
            old_data = data.get()
            updated_data = (
                new_data if old_data.empty else pd.concat([old_data, new_data])
            )
            data.set(updated_data)
            types = list(data()["type"].unique())
            times = ["days", "hours", "min"]
            concs = ["ug/L", "ng/L", "mg/L"]
            ui.update_select("data_influentID", choices=types, selected="")
            ui.update_select("data_effluentID", choices=types, selected="")
            ui.update_select("data_time_type", choices=times, selected="")
            ui.update_select("data_conc_type", choices=concs, selected="")

        else:
            cols = (
                ", ".join([f"{col!r}" for col in required_cols[:-1]])
                + f" and {required_cols[-1]!r}"
            )
            m = ui.modal(f"All of the columns {cols} must be present in data!")
            ui.modal_show(m)

    @reactive.effect
    @reactive.event(input.data_reset)
    def _():
        data.set(pd.DataFrame(columns=["type", "time", "concentration", "compound"]))
        ui.update_select("data_influentID", choices=[])
        ui.update_select("data_effluentID", choices=[])
        ui.update_select("data_time_type", selected="")
        ui.update_select("data_conc_type", selected="")


    @render.data_frame
    def infl_df():
        return render.DataTable(
            data().query(f"type=={input.data_influentID()!r}").drop("type", axis=1)
        )

    @render.data_frame
    def effl_df():
        return render.DataTable(
            data().query(f"type=={input.data_effluentID()!r}").drop("type", axis=1)
        )

    ####
    # SIMULATION
    #######

    @reactive.effect(priority=1)
    @reactive.event(input.run)
    async def _():
        nonlocal simulations
        scenario_sel = input.sel_run_scenario()
        
        # guard 1
        if not scenario_sel:
            m = ui.modal(
                "At least one scenario must be selected!"
                if len(scenarios()) > 0
                else "There are no scenarios to select from! Add one before proceeding"
            )
            ui.modal_show(m)

            if len(scenarios()) == 0:
                ui.update_navs("page", selected="input_tab")
                ui.update_navs("input_panels", selected="scenario_panel")
            return
        
        # guard 2
        msgs = []
        if data().empty:
            msgs.append("<b>Influent Data</b> can't be empty!<br>")
        for id, tag in zip(["data_influentID", "data_conc_type", "data_time_type"], ["Influent Keyword", "Concentration Unit", "Time Unit"]):
            if not input.__dict__["_map"][id]():
                msg = f"Please Specify <b>{tag}</b>!<br>"
                msgs.append(msg)
        if msgs:
            m = ui.modal(HTML("".join(["<h1>Insufficient Input!</h1>"]+ msgs)))
            ui.modal_show(m)
            ui.update_navs("page", selected="input_tab")
            ui.update_navs("input_panels", selected="data_panel")
            return

        # iterate over scenarios
        with ui.Progress(max=len(scenario_sel)) as p:
            for i, (scenario, scenario_data) in enumerate(
                {
                    key: val for key, val in scenarios().items() if key in scenario_sel
                }.items()
            ):
                p.set(i, message=f"Calculating {scenario!r}")
                scenario_adsorbates = scenario_data["adsorbates"]
                #if scenario in simulations:
                #     continue

                simulations.update({scenario: {"steps": {}}})

                # iterate over processing steps
                with ui.Progress(max=len(scenario_data["steps"])) as pstep:
                    for step, (_, step_data) in enumerate(
                        scenario_data["steps"].items()
                    ):
                        step = int(step)

                        # get step data
                        adsorbent_name = step_data["step_adsorbent"]
                        adsorbent = adsorbents().get(adsorbent_name)
                        column = columns().get(step_data["step_column"])

                        # subset compounds and adsprops
                        step_adsprops = {
                            key[key.find("::") + 2 :]: value
                            for key, value in adsprops().items()
                            if key.startswith(f"{adsorbent_name}::")
                            and key.endswith(tuple(scenario_adsorbates))
                        }

                        step_compounds = {
                            key: value
                            for key, value in compounds().items()
                            if key in step_adsprops.keys()
                        }

                        adsorbate_names = ", ".join(
                            [f"{name!r}" for name in list(step_compounds.keys())]
                        )

                        input_data = (
                            data().query(f"type == {input.data_influentID()!r}")
                            if step == 0
                            else last_output
                        )

                        # split input data whether its in adsprops or not
                        skipped_data = input_data.query(
                            f"compound not in {list(step_adsprops.keys())}"
                        )

                        input_data = input_data.query(
                            f"compound in {list(step_adsprops.keys())}"
                        )

                        input_data.to_excel("input_data.xlsx")

                        # update progressbar
                        pstep.set(
                            step,
                            message=f"Calculating Step {step}\n{adsorbent_name!r} in {step_data['step_column']!r}",
                        )

                        # match IX or GAC
                        match adsorbent["ads_type"]:
                            case "IX":
                                # convert to psdmix input
                                params, ions, Cin = make_psdmix_input(
                                    adsorbent,
                                    column,
                                    step_compounds,
                                    step_adsprops,
                                    input_data,
                                )

                                # save_inputs(
                                #    params, ions, Cin, scenario=scenario, step=step
                                # )

                                # check if compounds are present in influent data
                                if ions.index.intersection(Cin.columns).size == 0:
                                    msg = msgs["no_adsprops"].format(
                                        step=step,
                                        adsorbate_names=adsorbate_names,
                                        scenario=scenario,
                                        adsorbent_name=adsorbent_name,
                                    )
                                    ui.notification_show(
                                        HTML(msg), type="error", duration=None
                                    )
                                    model = None
                                    output = input_data
                                elif (not "BICARBONATE" in Cin.columns) and (
                                    not set(["ALKALINITY", "PH"]).issubset(Cin.columns)
                                ):
                                    msg = HTML(
                                        f"For IX, either 'BICARBONATE' or 'ALKALINITY' and 'PH' must be present in input data! <br>Skipping"
                                    )
                                    ui.notification_show(msg, type="warning")
                                    model = None
                                    output = input_data
                                else:
                                    # accepts file-like input (filename expected)
                                    with io.BytesIO() as buf:
                                        with pd.ExcelWriter(buf) as writer:
                                            for df, name in zip(
                                                [params, ions, Cin],
                                                ["params", "ions", "Cin"],
                                            ):
                                                df.to_excel(writer, sheet_name=name)
                                        model = PSDMIX(buf)
                                    model.solve()

                                    # channel output to df instead of file
                                    with io.BytesIO() as buf:
                                        model.save_results(buf)
                                        raw_output = pd.read_excel(buf)

                                    # raw_output.to_excel("raw.xlsx")
                                    output = make_psdmix_output(raw_output, column)

                            case "GAC":
                                # convert to psdm-inout
                                column_data, comp_data, rawdata_df, keywords = (
                                    make_psdm_input(
                                        adsorbent,
                                        column,
                                        step_compounds,
                                        step_adsprops,
                                        scenario_data,
                                        input_data,
                                    )
                                )

                                if keywords["k_data"].empty:
                                    msg = msgs["no_adsprops"].format(
                                        step=step,
                                        adsorbate_names=adsorbate_names,
                                        scenario=scenario,
                                        adsorbent_name=adsorbent_name,
                                    )
                                    ui.notification_show(HTML(msg), type="warning")
                                    model = None
                                    output = input_data
                                elif (
                                    keywords["k_data"]
                                    .columns.intersection(
                                        rawdata_df["influent"].columns
                                    )
                                    .size
                                    == 0
                                ):
                                    msg = msgs["no_adsprops"].format(
                                        step=step,
                                        adsorbate_names=adsorbate_names,
                                        scenario=scenario,
                                        adsorbent_name=adsorbent_name,
                                    )
                                    ui.notification_show(HTML(msg), type="warning")
                                    model = None
                                    output = input_data
                                    pass
                                else:
                                    # save_inputs(column_data, comp_data, rawdata_df, keywords["k_data"], scenario=scenario, step=step)
                                    # run model
                                    model = PSDM.PSDM(
                                        column_data=column_data,
                                        comp_data=comp_data,
                                        rawdata_df=rawdata_df,
                                        **keywords,
                                    )
                                    raw_output = model.run_psdm()

                                    if not raw_output:
                                        ui.notification_show(
                                            "Something went terribly wrong! Read console output for diagnostical information!",
                                            tyep="error",
                                            duration=None,
                                        )
                                        break
                                    output = make_psdm_output(raw_output, column)

                        # reattach data from "ignored" compounds
                        output = pd.concat([output, skipped_data])

                        # update simulation dict
                        simulation = {
                            step: {
                                "model": model,
                                "output": output,
                                "type": adsorbent["ads_type"],
                                "column": column
                            }
                        }
                        simulations[scenario]["steps"].update(simulation)

                        # funnel output in next step
                        last_output = make_new_input(output)
                        # last_output.to_excel("output.xlsx")

    @render.download(filename=lambda: f"{input.project_name()}.zip")
    async def sim_save():
        with io.BytesIO() as zipbuf:
            with ZipFile(zipbuf, mode="x") as zip:
                for simulation, sim_data in simulations.items():
                    if not simulation in input.sel_run_scenario():
                        continue
                    with io.BytesIO() as buf:
                        with pd.ExcelWriter(buf) as sheets:
                            for step, step_data in sim_data["steps"].items():
                                step_data["output"].to_excel(
                                    sheets, sheet_name=f"Step {step}"
                                )
                        safe_name = re.sub(r"[^\w_.)( -]", "_", simulation)
                        zip.writestr(f"{safe_name}.xlsx", buf.getvalue())
            yield zipbuf.getvalue()

    def save_inputs(*dfs, scenario, step):
        with pd.ExcelWriter(f"{scenario.replace('::','_')}_{step}.xlsx") as writer:
            for i, df in enumerate(dfs):
                df.to_excel(writer, sheet_name=str(i))

    def col_EBCT(column):
        ureg = UnitRegistry()
        L = ureg.Quantity(column["col_L"], column["col_L_unit"])
        d = ureg.Quantity(column["col_diam"], column["col_diam_unit"])
        V_dot = ureg.Quantity(column["col_flrt"], column["col_flrt_unit"])
        V_R = np.pi / 4 * d**2 * L
        EBCT = (V_R / V_dot).to(input.data_time_type())
        return EBCT

    def make_psdmix_output(rawdata, column):
        df = rawdata.rename(
            {input.data_time_type(): "time"},
            axis=1,
        )
        df = df.rename(
            {
                col: col.removesuffix(f" ({input.data_conc_type()})")
                for col in df.columns
            },
            axis=1,
        )
        df = df.melt(
            id_vars=["time"],
            var_name="compound",
            value_name="concentration",
        )

        EBCT = col_EBCT(column).magnitude
        df["BV"] = df.time / EBCT
        return df

    def make_psdm_output(rawdata, column):
        dfs = []
        for compound, out_data in rawdata.items():
            df = pd.DataFrame()
            # resample time axis
            x = np.linspace(out_data.x.min(), out_data.x.max(), 500)
            df["time"] = x
            df["concentration"] = out_data(x)
            dfs.append(df)
            df["compound"] = compound

        out = pd.concat(dfs)
        EBCT = col_EBCT(column).magnitude
        out["BV"] = out.time / EBCT
        return out

    def make_new_input(psdm_ix_output) -> pd.DataFrame:
        out = psdm_ix_output.copy()
        out["type"] = "influent"
        return out

    def make_psdm_input(
        adsorbent, column, compounds, adsprops, scenario_data, input_data
    ) -> tuple:
        column_data = make_column_data(adsorbent, column)
        comp_data = make_comp_data(compounds)
        k_data = make_k_data(adsprops, compounds)
        keywords = make_keywords(scenario_data, column, adsprops, k_data)
        raw_data = make_raw_data(input_data, column)

        return column_data, comp_data, raw_data, keywords

    def make_raw_data(rawdata, column) -> pd.DataFrame:
        rawdata = rawdata.copy().drop_duplicates(["time", "compound"])
        rawdata_df = (
            rawdata.pivot(
                columns=["type", "compound"], values=["concentration"], index=["time"]
            )
            .interpolate(method="index")
            .droplevel(0, axis=1)
            .sort_index(axis=1)
        )
        rawdata_df.rename(
            {input.data_effluentID(): column.get("col_name")},
            axis=1,
            inplace=True,
        )
        return rawdata_df

    def make_k_data(adsprops, compounds) -> pd.DataFrame:
        comp_data = [
            pd.DataFrame.from_dict(
                {
                    "K": convert_K(
                        val["adsprop_K"],
                        val["adsprop_n"],
                        val["adsprop_K_unit"],
                        "(ug/g)/((ug/L)**{n})",
                    ),
                    "1/n": val["adsprop_n"],
                },
                orient="index",
                columns=[key],
            )
            for key, val in adsprops.items()
            if key in compounds.keys()
        ]
        if all([df.empty for df in comp_data]):
            return pd.DataFrame()

        k_data = pd.concat(
            comp_data,
            axis=1,
        )
        return k_data

    def make_column_data(adsorbent, column) -> pd.Series:
        col_index = {
            "L": "cm",
            "diam": "cm",
            "wt": "g",
            "flrt": "ml/min",
            "epor": None,
            "rhop": "g/ml",
            "rhof": "g/ml",
            "rad": "cm",
            "tortu": None,
            "psdfr": None,
            "influentID": None,
            "effluentID": None,
            "units": None,
        }
        source = {**adsorbent, **column}
        df_dict = reorder_dict_and_convert_units(source, col_index)
        column_data = pd.DataFrame.from_dict(
            df_dict, orient="index", columns=[column.get("col_name")]
        ).squeeze()

        volume = np.pi / 4 * column_data.diam**2 * column_data.L
        column_data["wt"] = volume * column_data.rhop
        column_data["influentID"] = input.data_influentID()
        column_data["effluentID"] = input.data_effluentID()

        return column_data

    def make_comp_data(compounds) -> pd.DataFrame:
        comp_index = {
            "MW": "g/mol",
            "MolarVol": "mol/L",
            "BP": "degreeC",
            "Density": "g/ml",
        }

        comp_data = pd.concat(
            [
                pd.DataFrame.from_dict(
                    reorder_dict_and_convert_units(data, comp_index),
                    orient="index",
                    columns=[compound],
                )
                for compound, data in compounds.items()
            ],
            axis=1,
        )
        return comp_data

    def make_keywords(scenario_data, column, adsprops, k_data):
        mass_transfer = {
            key: {
                "kf": value["adsprop_kf"],
                "dp": value["adsprop_dp"],
                "ds": value["adsprop_ds"],
            }
            for key, value in adsprops.items()
        }

        keywords = {
            "nr": input.sim_nr(),
            "nz": input.sim_nz(),
            "ne": int(input.sim_ne()),
            "temp": column.get("col_temp", 20),
            "time_type": input.data_time_type(),
            "conc_type": input.data_conc_type().removesuffix("/L"),
            "water_type": scenario_data["water_type"],
            "chem_type": scenario_data["chem_type"],
            "mass_transfer": mass_transfer,
            "k_data": k_data,
        }

        return keywords

    def reorder_dict_and_convert_units(source, target):
        ureg = UnitRegistry()
        cd = {
            key[key.find("_") + 1 :]: {
                "val": value,
                "unit": (source[f"{key}_unit"] if f"{key}_unit" in source else None),
            }
            for key, value in source.items()
            if key.endswith(tuple(target.keys()))
        }
        # unit conversion
        factor = {
            key: (ureg(value["unit"]).to(target[key]).magnitude)
            for key, value in cd.items()
            if value["unit"]
        }
        return {
            key: value["val"] * factor.get(key) if factor.get(key) else value["val"]
            for key, value in cd.items()
        }

    def make_psdmix_input(adsorbent, column, compounds, adsprops, input_data):
        params = make_params(adsorbent, column)
        ions = make_ions(compounds, adsprops)
        Cin = make_Cin(input_data)
        return params, ions, Cin

    def make_ions(ions, adsprops) -> pd.DataFrame:
        dfs = []
        for ion, values in ions.items():
            if ion not in adsprops:
                continue
            df = pd.Series(name=ion)
            df["mw"] = values["comp_MW"]
            df["Kxc"] = adsprops[ion]["adsprop_K"]
            df["valence"] = values["comp_valence"]
            unit = input.data_conc_type()
            df["units"] = unit.removesuffix("/L")
            dfs.append(df)
        if dfs:
            out = pd.concat(dfs, axis=1).T
            out.index.rename("name", inplace=True)
            return out
        else:
            return pd.DataFrame()

    def make_params(adsorbent, column):
        d = [
            {
                "name": key[key.find("_") + 1 :],
                "value": value,
                "units": adsorbent.get(f"{key}_unit", pd.NA),
            }
            for key, value in adsorbent.items()
            if not key.endswith("_unit")
        ]
        df = pd.DataFrame.from_records(d, index="name")
        d = [
            {
                "name": "EBED",
                "value": 1 - adsorbent["ads_rhop"] / adsorbent["ads_rhof"],
                "units": pd.NA,
            },
            {
                "name": "flrt",
                "value": column["col_flrt"],
                "units": column["col_flrt_unit"],
            },
            {
                "name": "L",
                "value": column["col_L"],
                "units": column["col_L_unit"],
            },
            {"name": "nr", "value": input.sim_nr(), "units": pd.NA},
            {"name": "nz", "value": input.sim_nz(), "units": pd.NA},
            {"name": "time", "value": 1, "units": input.data_time_type()},
            {
                "name": "diam",
                "value": column["col_diam"],
                "units": column["col_diam_unit"],
            },
        ]
        df = pd.concat([df, pd.DataFrame.from_records(d, index="name")])

        mapper = {
            "rad": "rb",
            "epor": "EPOR",
            "Q": "Qf",
            "kf": "kL",
            "dp": "Ds",
            "ds": "Dp",
        }
        return df.rename(mapper).drop(
            [
                "type",
                "rhof",
                "rhop",
                "psdfr",
                "tortu",
                "name",
            ]
        )

    def make_Cin(rawdata):
        df = (
            rawdata.pivot(
                columns=["type", "compound"], values=["concentration"], index=["time"]
            )
            .interpolate(method="index")
            .droplevel(0, axis=1)
            .sort_index(axis=1)
        )[input.data_influentID()]
        df.index.rename("Time", inplace=True)
        return df

    @render_widget
    # @reactive.event(input.run)
    def plot_results():
        dfs = []

        # match real_data to every scenario
        real_data = data()

        for simulation, sim_data in simulations.items():
            for step, step_data in sim_data["steps"].items():
                df = step_data["output"].copy()
                df["step"] = step
                df["scenario"] = simulation
                df["type"] = "effluent_model"
                df_real = real_data[
                    real_data.step
                    == step
                    & ((real_data.scenario == simulation) | (real_data.scenario.isna()))
                ]
                df_real.loc[df_real.scenario.isna(), ["scenario"]] = simulation
                column = step_data["column"]
                EBCT = col_EBCT(column).magnitude
                df_real["BV"] = df_real.time / EBCT
                df = pd.concat([df, df_real])
                dfs.append(df)

        # combine influent and effluent
        if not dfs:
            return
        out = pd.concat(dfs)
        plot = px.line(
            out,
            x=input.simplot_x_axis(),
            y="concentration",
            color="compound",
            facet_col="scenario",
            facet_row="step",
            line_dash="type",
            height=int(GetSystemMetrics(1) * 0.7),
            markers=input.simplot_marker(),
            labels={
                "time": input.data_time_type(),
                "concentration": input.data_conc_type(),
            },
        )
        return plot


app = App(app_ui, server)

# app.run(launch_browser=True)
