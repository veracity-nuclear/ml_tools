from typing import Dict, List, Optional
import pylab as plt
import numpy as np
import time

from ml_tools.model.state import StateSeries
from ml_tools.model.prediction_strategy import PredictionStrategy

def plot_ref_vs_pred(models:                   Dict[str, PredictionStrategy],
                      state_series:            List[StateSeries],
                      fig_name:                str,
                      index:                   int = -1,
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
    index : int
        The index of the state in the series to be plotted (Default: -1)
    fig_name : str
        A name for the figure that is generated
    error_bands : List[float]
        The Error (i.e.Tolerance) Bands to be plotted in terms of ±X% (Default: [2.5, 5.0])
    title : bool
        Flag for whether or not a title should be included on the figure
    predicted_feature_label : Optional[str]
        The label to use for the predicted feature (Default: predicted feature label from models)
    """

    plt.figure(figsize=(10,6))

    predicted_feature = next(iter(models.values())).predicted_feature
    reference         = np.asarray([series[index].feature(predicted_feature) for series in state_series])
    for label, model in models.items():
        assert model.isTrained
        assert model.predicted_feature == predicted_feature
        predicted = np.asarray([series[index] for series in model.predict(state_series)])
        plt.plot(reference, predicted, '.', alpha=0.3, label=label)

    max_val = max(*plt.xlim(), *plt.ylim())
    pylab.axis([0, max_val, 0, max_val])

    alpha = 1.0
    plt.plot([0, hi], [0, hi], '--k', alpha=alpha, label='Reference')
    for band in error_bands:
        alpha -= 1./(len(error_bands) + 1.)
        plt.plot([band, max_val+5+band, max_val+5-band, 0], [0, max_val+5, max_val+5, band],
                 '--k', alpha=alpha, label='+/- {0:3.1f}'.format(band))

    plt.grid(True)
    predicted_feature_label = predicted_feature if predicted_feature_label is None else predicted_feature_label
    plt.xlabel('Reference ' + predicted_feature_label, fontsize=14)
    plt.ylabel('Predicted ' + predicted_feature_label, fontsize=14)
    if title:
        plt.title('Reference vs. Predicted ' + predicted_feature_label, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14)
    plt.savefig(fig_name, dpi=600, bbox_inches='tight')
    plt.close()




def plot_hist(models:       Dict[str, PredictionStrategy],
              state_series: List[StateSeries],
              fig_name:     str,
              index:        int = -1,
              title:                   bool = True,
              predicted_feature_label: Optional[str] = None) -> None:
    """ Function for plotting reference minus predicted histograms of a collection of models

    Parameters
    ----------
    models : Dict[str, PredictionStrategy]
        The collection of models (i.e. prediction strategies) whose predictions are to be plotted.
        The dictionary key will be the label used for the model in the plot legend
    state_series : List[StateSeries]
        The state series to use for plotting
    index : int
        The index of the state in the series to be plotted (Default: -1)
    fig_name : str
        A name for the figure that is generated
    title : bool
        Flag for whether or not a title should be included on the figure
    predicted_feature_label : Optional[str]
        The label to use for the predicted feature (Default: predicted feature label from models)
    """

    diffs             = []
    maxdiff           = 0.
    predicted_feature = next(iter(models.values())).predicted_feature
    reference         = np.asarray([series[index].feature(predicted_feature) for series in state_series])
    for label, model in models.items():
        assert model.isTrained
        assert model.predicted_feature == predicted_feature
        predicted = np.asarray([series[index] for series in model.predict(state_series)])
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
    plt.savefig(fig_name)


def print_metrics(models: Dict[str, PredictionStrategy], state_series: List[StateSeries], index: int = -1,) -> None:
    """ Function for printing the statistical metrics (average, std.dev., rms, max) of models to the screen

    Parameters
    ----------
    models : Dict[str, PredictionStrategy]
        The collection of models (i.e. prediction strategies) whose predictions are to be plotted.
        The dictionary key will be the label used for the model in the plot legend
    state_series : List[StateSeries]
        The state series to use for plotting
    index : int
        The index of the state in the series to be plotted (Default: -1)
    """

    predicted_feature = next(iter(models.values())).predicted_feature
    reference         = np.asarray([series[index].feature(predicted_feature) for series in state_series])

    padding = np.max([len(label) for label in models.keys()]) + 3
    fmtstr  = '{{0:{:d}s}} {{1:8.5f}} {{2:7.5f}} {{3:7.5f}} {{4:7.5f}} {{5:9.3e}}'.format(padding)
    print(' ' * padding + '   Avg     Std     RMS     Max   Time/State')
    for label, model in models.items():
        start = time.time()
        predicted = np.asarray([series[index] for series in model.predict(state_series)])
        dt = time.time() - start
        diff = reference - predicted
        rms = np.sqrt(np.dot(diff.flatten(),diff.flatten()))/float(len(diff))
        max = np.max(np.abs(diff))
        avg = np.mean(diff)
        std = np.std(diff)
        print(fmtstr.format(label + ' :', avg, std, rms, max, dt/len(state_series)))
