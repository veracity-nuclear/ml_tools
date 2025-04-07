from typing import Dict, List, Optional
import time
from copy import deepcopy

import pylab as plt
import numpy as np
import seaborn as sns
import pandas as pd

from ml_tools.model.state import State, StateSeries, series_to_pandas
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.model.feature_perturbator import FeaturePerturbator


def plot_ref_vs_pred(models:                  Dict[str, PredictionStrategy],
                     state_series:            List[StateSeries],
                     fig_name:                str = 'ref_vs_pred',
                     state_index:             int = -1,
                     array_index:             int = 0,
                     error_bands:             List[float] = [2.5, 5.0],
                     title:                   bool = True,
                     predicted_feature_label: Optional[str] = None) -> None:
    """ Function for plotting reference vs. predicted results of a collection of models

    Parameters
    ----------
    models : Dict[str, PredictionStrategy]
        The collection of models (i.e. prediction strategies) whose predictions are to be plotted.
        The dictionary key will be the label used for the model in the plot legend
    state_series : List[StateSeries]
        The collection of state series to use for plotting
    state_index : int
        The index of the state in the series to be plotted (Default: -1)
    array_index : int
        The index of the predicted value array to be plotted (Default: 0)
    fig_name : str
        A name for the figure that is generated
    error_bands : List[float]
        The Error (i.e. Tolerance) Bands to be plotted in terms of ±X% (Default: [2.5, 5.0])
    title : bool
        Flag for whether or not a title should be included on the figure
    predicted_feature_label : Optional[str]
        The label to use for the predicted feature (Default: predicted feature label from models)
    """

    plt.figure(figsize=(10,6))

    predicted_feature = next(iter(models.values())).predicted_feature
    reference         = np.asarray([series[state_index][predicted_feature][array_index]
                          for series in state_series])
    for label, model in models.items():
        assert model.isTrained
        assert model.predicted_feature == predicted_feature
        predicted = np.asarray([series[state_index][array_index] for series in model.predict(state_series)])
        plt.plot(reference, predicted, '.', alpha=0.3, label=label)

    max_val = max(*plt.xlim(), *plt.ylim())
    plt.axis([0, max_val, 0, max_val])

    alpha = 1.0
    plt.plot([0, max_val], [0, max_val], '--k', alpha=alpha, label='Reference')
    for band in error_bands:
        alpha -= 1./(len(error_bands) + 1.)
        plt.plot([band, max_val+5+band, max_val+5-band, 0], [0, max_val+5, max_val+5, band],
                 '--k', alpha=alpha, label=f'+/- {band:3.1f}')

    plt.grid(True)
    predicted_feature_label = predicted_feature if predicted_feature_label is None else predicted_feature_label
    plt.xlabel('Reference ' + predicted_feature_label, fontsize=14)
    plt.ylabel('Predicted ' + predicted_feature_label, fontsize=14)
    if title:
        plt.title('Reference vs. Predicted ' + predicted_feature_label, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14)
    plt.savefig(fig_name+'.png', dpi=600, bbox_inches='tight')
    plt.close()




def plot_hist(models:       Dict[str, PredictionStrategy],
              state_series: List[StateSeries],
              fig_name:     str = 'hist',
              state_index:  int = -1,
              array_index:  int = 0,
              predicted_feature_label: Optional[str] = None) -> None:
    """ Function for plotting reference minus predicted histograms of a collection of models

    Parameters
    ----------
    models : Dict[str, PredictionStrategy]
        The collection of models (i.e. prediction strategies) whose predictions are to be plotted.
        The dictionary key will be the label used for the model in the plot legend
    state_series : List[StateSeries]
        The state series to use for plotting
    state_index : int
        The index of the state in the series to be plotted (Default: -1)
    array_index : int
        The index of the predicted value array to be plotted (Default: 0)
    fig_name : str
        A name for the figure that is generated
    predicted_feature_label : Optional[str]
        The label to use for the predicted feature (Default: predicted feature label from models)
    """

    diffs             = []
    maxdiff           = 0.
    predicted_feature = next(iter(models.values())).predicted_feature
    reference         = np.asarray([series[state_index][predicted_feature][array_index]
                            for series in state_series])
    for label, model in models.items():
        assert model.isTrained
        assert model.predicted_feature == predicted_feature
        predicted = np.asarray([series[state_index][array_index] for series in model.predict(state_series)])
        diffs.append(reference - predicted)
        maxdiff = max(maxdiff, np.max(np.abs(diffs[-1])))

    bins = np.linspace(-maxdiff, maxdiff, 100, endpoint=True)
    for label, diff in zip(models.keys(), diffs):
        plt.hist(diff, bins, alpha=0.3, label=label)

    plt.grid(True)
    predicted_feature_label = predicted_feature if predicted_feature_label is None else predicted_feature_label
    plt.xlabel('Reference - Predicted ' + predicted_feature_label)
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(fig_name+'.png')
    plt.close()


def plot_sensitivities(models:                  Dict[str, PredictionStrategy],
                       state_series:            List[StateSeries],
                       perturbators:            List[FeaturePerturbator],
                       number_of_perturbations: int,
                       state_index:             int = -1,
                       array_index:             int = 0,
                       fig_name_prefix:         str = 'box_plot',
                       num_procs:               int = 1):

    """ Function for creating state perturbation sensitivity box plots of models

    This function perturbs the whole series of states when performing perturbations

    Parameters
    ----------
    models : Dict[str, PredictionStrategy]
        The collection of models (i.e. prediction strategies) whose sensitivities are to be plotted.
        The dictionary key will be the suffix of the figure file name
    state_series : List[StateSeries]
        The collection of state series to use for evaluating the sensitivities
    perturbators : List[FeaturePerturbator]
        The perturbators to use for perturbing the state features
    number_of_perturbations : int
        The number of perturbation realizations to perform
    state_index : int
        The index of the state in the series to be plotted (Default: -1)
    array_index : int
        The index of the predicted value array to be plotted (Default: 0)
    fig_name_prefix : str
        The prefix for the figure files that will be created
    num_procs : int
        The number of parallel processors to use when perturbing states
    """

    perturbations = []
    for _ in range(number_of_perturbations):
        perturbations.append([])
        for series in state_series:
            perturbations[-1].append(State.perturb_states(perturbators, series, num_procs))

    predicted_feature = next(iter(models.values())).predicted_feature

    for label, model in models.items():
        assert model.isTrained
        assert model.predicted_feature == predicted_feature

        predicted = np.asarray([series[state_index][array_index] for series in model.predict(state_series)])
        results   = [[] for series in state_series]
        for perturbation in perturbations:
            perturbed_predicted = model.predict(perturbation)
            for i in range(len(state_series)):
                results[i].append(perturbed_predicted[i][state_index][array_index])

        order = np.argsort(predicted)
        plt.figure(figsize=(12, 5))
        plt.boxplot([results[i] for i in order])
        plt.plot(range(1,len(order)+1), [predicted[i] for i in order],'.r')
        ticks = plt.gca().get_xticks()
        plt.xticks(ticks, [str(int(tick)) if int(tick) % 10 == 0 else '' for tick in ticks])
        plt.savefig(fig_name_prefix+"_"+label+'.png', dpi=600, bbox_inches='tight')
        plt.close()



def print_metrics(models:       Dict[str, PredictionStrategy],
                  state_series: List[StateSeries],
                  state_index:  int = -1,
                  array_index:  int = 0,) -> None:
    """ Function for printing the statistical metrics (average, std.dev., rms, max) of models to the screen

    Parameters
    ----------
    models : Dict[str, PredictionStrategy]
        The collection of models (i.e. prediction strategies) whose predictions are to be plotted.
        The dictionary key will be the label used for the model in the plot legend
    state_series : List[StateSeries]
        The state series to use for plotting
    state_index : int
        The index of the state in the series to be plotted (Default: -1)
    array_index : int
        The index of the predicted value array to be plotted (Default: 0)
    """

    predicted_feature = next(iter(models.values())).predicted_feature
    reference         = np.asarray([series[state_index][predicted_feature][array_index]
                            for series in state_series])

    padding = np.max([len(label) for label in models.keys()]) + 3
    fmtstr  = f'{{0:{padding}s}} {{1:8.5f}} {{2:7.5f}} {{3:7.5f}} {{4:7.5f}} {{5:9.3e}}'
    print(' ' * padding + '   Avg     Std     RMS     Max   Time/State')
    for label, model in models.items():
        start = time.time()
        predicted = np.asarray([series[state_index][array_index] for series in model.predict(state_series)])
        dt = time.time() - start
        diff = reference - predicted
        rms = np.sqrt(np.dot(diff.flatten(),diff.flatten()))/float(len(diff))
        maxdiff = np.max(np.abs(diff))
        avg = np.mean(diff)
        std = np.std(diff)
        print(fmtstr.format(label + ' :', avg, std, rms, maxdiff, dt/len(state_series)))


def plot_corr_matrix(input_features:  List[str],
                     state_series:    List[StateSeries],
                     state_index:     int = -1,
                     fig_name:        str = 'corr_matrix') -> None:
    """ Function for plotting the correlation matrix of a given set of input features

    Parameters
    ----------
    input_features : List[str]
        A list specifying the input features whose correlations are to be plotted.
    state_series : List[StateSeries]
        The state series to use for plotting
    state_index : int
        The index of the state in the series to be plotted (Default: -1)
    fig_name : str
        A name for the figure that is generated (Default: 'corr_matrix')
    """

    X = series_to_pandas(state_series, input_features)

    if state_index < 0:
        max_index = X.index.get_level_values('state_index').max()
        state_index = max_index + 1 + state_index

    valid_indices = X.index.get_level_values('state_index').unique()
    assert state_index in valid_indices, \
        f"state_index {state_index} is out of bounds (valid: {valid_indices.min()} to {valid_indices.max()})"

    input_features = [col for col in X.columns if any(col == prefix or col.startswith(prefix) for prefix in input_features)]

    X           = X.loc[(slice(None), state_index), :]
    X           = X[input_features]
    corr_matrix = X.corr()
    corr_matrix = corr_matrix.fillna(0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, xticklabels=input_features, yticklabels=input_features, cmap='coolwarm', fmt='.2f')
    plt.tight_layout()
    plt.savefig(fig_name+'.png', dpi=600, bbox_inches='tight')
    plt.close()


def plot_ice_pdp(models:          Dict[str, PredictionStrategy],
                 state_series:    List[StateSeries],
                 input_feature:   str,
                 fig_name_prefix: str = 'ice_pdp',
                 state_index:     int = -1,
                 input_index:     int = 0,
                 output_index:    int = 0,
                 num_points:      int = 50,
                 predicted_feature_label: Optional[str] = None) -> None:
    """ Function to plot ICE/PDP feature analyses for a given set of models.
    Parameters
    ----------
    models : Dict[str, PredictionStrategy]
        The collection of models (i.e. prediction strategies) whose sensitivities are to be plotted.
        The dictionary key will be the suffix of the figure file name
    state_series : List[StateSeries]
        The state series to use for plotting
    input_feature : str
        The list of features to be included in SHAP analysis
    fig_name_prefix : str, optional
        The prefix for the figure files that will be created (Default: 'ice_pdp')
    state_index : int, optional
        The index of the state in the series to be analyzed (Default: -1, last state)
    input_index : int
        The index of the input feature value array to be plotted (Default: 0)
    output_index : int
        The index of the predicted value array to be plotted (Default: 0)
    num_points : int, optional
        The number of sampled points for ICE and PDP curves (Default: 50)
    predicted_feature_label : Optional[str]
        The label to use for the predicted feature (Default: predicted feature label from models)
    """

    predicted_feature = next(iter(models.values())).predicted_feature

    values = np.asarray([series[state_index][input_feature][input_index] for series in state_series])
    values = np.unique(values)
    values = np.random.choice(values, size=min(num_points, len(values)), replace=False)
    values.sort()

    for label, model in models.items():
        assert model.isTrained, f"Model {label} must be trained before ICE/PDP analysis."

        ice_data = []
        for value in values:
            state_series_perturbed = deepcopy(state_series)
            for series in state_series_perturbed:
                series[state_index][input_feature][input_index] = value

            predictions = model.predict(state_series_perturbed)
            for i, series in enumerate(predictions):
                ice_data.append({'sample': i,
                                 'value': value,
                                 'prediction': series[state_index][output_index]})

            ice_df = pd.DataFrame(ice_data)

        # ICE Curves
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=ice_df, x='value', y='prediction', hue='sample',
                     estimator=None, alpha=0.3, legend=False, linewidth=1)

        # PDP Curve
        pdp_df = ice_df.groupby('value')['prediction'].mean().reset_index()
        sns.lineplot(data=pdp_df, x='value', y='prediction', color='red', label='PDP', linewidth=2)

        # Rug plot for original (unperturbed) values
        sns.rugplot(x=values, height=0.03, color='black', alpha=0.3)

        # Axis labels
        predicted_feature_label = predicted_feature if predicted_feature_label is None else predicted_feature_label
        plt.xlabel(input_feature)
        plt.ylabel(predicted_feature_label)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{fig_name_prefix}_{label}_{input_feature}.png", dpi=600, bbox_inches='tight')
        plt.close()
