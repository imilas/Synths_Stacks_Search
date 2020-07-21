from kaleido.scopes.plotly import PlotlyScope
import optuna
import plotly
import plotly.express as px
import pandas as pd
from optuna.visualization.utils import _is_log_scale
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
from typing import Any
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional

from optuna.logging import get_logger
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import TrialState
from optuna.trial import FrozenTrial
# study = optuna.create_study(study_name='Study_ALL',sampler=optuna.samplers.TPESampler(), 
#                             pruner=optuna.pruners.HyperbandPruner(), storage='sqlite:///optuna_median.db',load_if_exists=True)

def get_parallel_coordinate_plot(study=None,coloring="blues",params= None,objective_value="Loss"):

    layout = go.Layout(title="Parallel Coordinate Plot",)

    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]

    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")
        return go.Figure(data=[], layout=layout)

    all_params = {p_name for t in trials for p_name in t.params.keys()}
    if params is not None:
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError("Parameter {} does not exist in your study.".format(input_p_name))
        all_params = set(params)
    sorted_params = sorted(list(all_params))

    dims = [
        {
            "label": objective_value,
            "values": tuple([t.value for t in trials]),
            "range": (round(min([t.value for t in trials]),3), round(max([t.value for t in trials]),3)),
        }
    ]  # type: List[Dict[str, Any]]
    for p_name in sorted_params:
        values = []
        for t in trials:
            if p_name in t.params:
                values.append(t.params[p_name])
        is_categorical = False
        try:
            tuple(map(float, values))
        except (TypeError, ValueError):
            vocab = defaultdict(lambda: len(vocab))  # type: DefaultDict[str, int]
            values = [vocab[v] for v in values]
            is_categorical = True
        dim = {
            "label": p_name if len(p_name) < 12 else "{}.".format(p_name[:17]),
            "values": tuple(values),
            "range": ( round(min(values),2), round(max(values),2)),
        }
        
        if is_categorical:
            dim["tickvals"] = list(range(len(vocab)))
            ticktext=list(sorted(vocab.items(), key=lambda x: x[1]))
            dim["ticktext"] = [x[0] for x in ticktext]
        if p_name in (["Time Steps", "Latent Size", "Frequency Bins"]):
            dim["tickvals"]= list(set(values))
        dims.append(dim)
    
    traces = [
        go.Parcoords(
            dimensions=dims,
            tickfont=plotly.graph_objs.parcoords.Tickfont(color="black",size=15,family="Times New Roman"),
            rangefont=plotly.graph_objs.parcoords.Rangefont(color="white",size=1,family="Times New Roman"),
            line={
                "color": dims[0]["values"],
                
                "colorscale": coloring,
                "colorbar": {"title": objective_value},
                "showscale": True,
#                 "reversescale":True,
            },
        )
    ]

    figure = go.Figure(data=traces, layout=layout)
    
    return figure



def get_slice_plot(study= Study,objective_value="Loss",params= None):

    layout = go.Layout(title="Slice Plot",)

    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]

    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")
        return go.Figure(data=[], layout=layout)

    all_params = {p_name for t in trials for p_name in t.params.keys()}
    if params is None:
        sorted_params = sorted(list(all_params))
    else:
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError("Parameter {} does not exist in your study.".format(input_p_name))
        sorted_params = sorted(list(set(params)))

    n_params = len(sorted_params)

    if n_params == 1:
        figure = go.Figure(
            data=[_generate_slice_subplot(study, trials, sorted_params[0])], layout=layout
        )
        figure.update_xaxes(title_text=sorted_params[0])
        figure.update_yaxes(title_text=objective_value)
        if _is_log_scale(trials, sorted_params[0]):
            figure.update_xaxes(type="log")
    else:
        figure = make_subplots(rows=1, cols=len(sorted_params), shared_yaxes=True)
        figure.update_layout(layout)
        showscale = True  # showscale option only needs to be specified once.
        for i, param in enumerate(sorted_params):
            trace = _generate_slice_subplot(study, trials, param)
            trace.update(marker={"showscale": showscale})  # showscale's default is True.
            if showscale:
                showscale = False
            figure.add_trace(trace, row=1, col=i + 1)
            if len(param)>10:

                if "L2" in param:
                    param="L2 Regular."
            figure.update_xaxes(title_text=param, row=1, col=i + 1)
            if i == 0:
                figure.update_yaxes(title_text=objective_value, row=1, col=1)
            if _is_log_scale(trials, param):
                figure.update_xaxes(type="log", row=1, col=i + 1)
        if n_params > 3:
            # Ensure that each subplot has a minimum width without relying on autusizing.
            figure.update_layout(width=300 * n_params)

    return figure


def _generate_slice_subplot(study: Study, trials: List[FrozenTrial], param: str) -> "Scatter":

    return go.Scatter(
        x=[t.params[param] for t in trials if param in t.params],
        y=[t.value for t in trials if param in t.params],

        mode="markers",
        marker={
            "line": {"width": 0.65, "color": "grey",},
            "color": [t.number for t in trials if param in t.params],
#             "reversescale":True,

            "colorbar": {
                "title": "# of Trials",
                "x": 1.0,  # Offset the colorbar position with a fixed width `xpad`.
                "xpad": 15,
            },
        },
        
        showlegend=False,
    )
