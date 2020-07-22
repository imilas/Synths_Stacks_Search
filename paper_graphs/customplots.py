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


from collections import OrderedDict
from typing import List
from typing import Optional

import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.importance._base import BaseImportanceEvaluator


# study = optuna.create_study(study_name='Study_ALL',sampler=optuna.samplers.TPESampler(), 
#                             pruner=optuna.pruners.HyperbandPruner(), storage='sqlite:///optuna_median.db',load_if_exists=True)
def name_shortner(p_name):
        if "L2" in p_name:
            p_name="L2 Regular."
        if "Drop" in p_name:
            p_name="Dropout"
        if "Freq" in p_name:
            p_name="Freq. Bins"
        if "Hidden" in p_name:
            p_name="Hidden"
        if "Learning" in p_name:
            p_name="Learn Rate"
        
        return p_name
def get_parallel_coordinate_plot(study=None,coloring="blues",params= None,objective_value="Loss",logLoss=True):

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
    
    if logLoss==True:
        for t in trials:
            t.value=-1*np.log(t.value)
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
            "label": name_shortner(p_name),
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
            
            tickfont=plotly.graph_objs.parcoords.Tickfont(color="black",size=18,family="Times New Roman"),
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

            figure.update_xaxes(title_text=name_shortner(param), row=1, col=i + 1, )
            if i == 0:
                figure.update_yaxes(title_text=objective_value, row=1, col=1)
            if _is_log_scale(trials, param):
                figure.update_xaxes(type="log", row=1, col=i + 1)
        if n_params > 3:
            # Ensure that each subplot has a minimum width without relying on autosizing.
            figure.update_layout(width=300 * n_params)
    figure.update_xaxes(ticks="outside")
    return figure


def _generate_slice_subplot(study: Study, trials: List[FrozenTrial], param: str) -> "Scatter":

    fig = go.Scatter(
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

    return fig



def get_optimization_history_plot(study: Study) -> "go.Figure":

    layout = go.Layout(
        title="Optimization History Plot",
        xaxis={"title": "#Trials"},
        yaxis={"title": "Objective Value"},
    )

    trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if len(trials) == 0:
        _logger.warning("Study instance does not contain trials.")
        return go.Figure(data=[], layout=layout)

    best_values = [float("inf")] if study.direction == StudyDirection.MINIMIZE else [-float("inf")]
    comp = min if study.direction == StudyDirection.MINIMIZE else max
    for trial in trials:
        trial_value = trial.value
        assert trial_value is not None  # For mypy
        best_values.append(comp(best_values[-1], trial_value))
    best_values.pop(0)
    traces = [
                go.Scatter(
            
            x=[t.number for t in trials],
            y=[t.value for t in trials],
            mode="markers",
            marker_color='rgb(11,11,255,0.1)',
            name="Objective Value",
            marker=dict(size=12,symbol='line-ns',line=dict(color='rgb(110,110,220,1)',width=2))
          
        ),
                go.Scatter(x=[t.number for t in trials],y=best_values, name="Best Value",     
                  marker_color='rgb(10,10,50)',
                   line = dict( width=2, ),
                   marker=dict(size=20,)
                   
                  ),


    ]

    figure = go.Figure(data=traces, layout=layout,)

    return figure


Blues = plotly.colors.sequential.ice_r
_distribution_colors = {
        UniformDistribution: Blues[-1],
        LogUniformDistribution: Blues[-1],
        DiscreteUniformDistribution: Blues[-1],
        IntUniformDistribution: Blues[-1],
        IntLogUniformDistribution: Blues[-1],
        CategoricalDistribution: Blues[-1],}

logger = get_logger(__name__)


def plot_param_importances(
    importance=None,study=None,
) -> "go.Figure":


    layout = go.Layout(
        title="Hyperparameter Importances",
        xaxis={"title": "Importance"},
        yaxis={"title": "Hyperparameter"},
        showlegend=False,
    )

    # Importances cannot be evaluated without completed trials.
    # Return an empty figure for consistency with other visualization functions.
    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if len(trials) == 0:
        logger.warning("Study instance does not contain completed trials.")
        return go.Figure(data=[], layout=layout)

    importances = importance
    print(importances)
    importances = OrderedDict(reversed(list(importances.items())))
    importance_values = list(importances.values())
    param_names = list(importances.keys())

    fig = go.Figure(
        data=[
            go.Bar(
                x=importance_values,
                y=param_names,
                text=importance_values,
                texttemplate="%{text:.2f}",
                textposition="outside",
                cliponaxis=False,  # Ensure text is not clipped.
                hovertemplate=[
                    _make_hovertext(param_name, importance, study)
                    for param_name, importance in importances.items()
                ],
                marker_color=[_get_color(param_name, study) for param_name in param_names],
                orientation="h",
            )
        ],
        layout=layout,
    )

    return fig



def _get_distribution(param_name: str, study: Study) -> BaseDistribution:
    for trial in study.trials:
        if param_name in trial.distributions:
            return trial.distributions[param_name]
    assert False


def _get_color(param_name: str, study: Study) -> str:
    return _distribution_colors[type(_get_distribution(param_name, study))]


def _make_hovertext(param_name: str, importance: float, study: Study) -> str:
    return "{} ({}): {}<extra></extra>".format(
        param_name, _get_distribution(param_name, study).__class__.__name__, importance
    )


def get_intermediate_plot(study: Study,topx=10,num_trials_threshold=30) -> "go.Figure":

    layout = go.Layout(
        title="Intermediate Values Plot",
        xaxis={"title": "Step"},
        yaxis={"title": "Intermediate Value"},

        
    )
    #this value will determine which trials we want
    df=study.trials_dataframe()
    v=df.sort_values(by=["value"],ascending=True).reset_index().iloc[topx]["value"]

    target_state = [TrialState.PRUNED, TrialState.COMPLETE, TrialState.RUNNING]
    trials = [trial for trial in study.trials if (trial.state in target_state) 
              and (trial.value and trial.value<v) and len(trial.intermediate_values)<num_trials_threshold]

    if len(trials) == 0:
        _logger.warning("Study instance does not contain trials.")
        return go.Figure(data=[], layout=layout)

    traces = []

    for i,trial in enumerate(trials):
        
        if trial.intermediate_values:
            sorted_intermediate_values = sorted(trial.intermediate_values.items())
            trace = go.Scatter(
                x=tuple((x for x, _ in sorted_intermediate_values)),
                y=tuple((y for _, y in sorted_intermediate_values)),
                mode="lines+markers",
                marker={"maxdisplayed": 10},
                marker_symbol=i,
                marker_size=8, 
                name="Trial{}".format(trial.number),
            )
            traces.append(trace)

    if not traces:
        _logger.warning(
            "You need to set up the pruning feature to utilize `plot_intermediate_values()`"
        )
        return go.Figure(data=[], layout=layout)

    figure = go.Figure(data=traces, layout=layout,)

    return figure