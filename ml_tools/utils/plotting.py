from typing import Dict, List, Optional, Any
import time
from copy import deepcopy
import os

import pylab as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import ray
import seaborn as sns
import shap

from ml_tools.model.state import State, SeriesCollection
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.model.feature_perturbator import FeaturePerturbator
from ml_tools.utils.status_bar import StatusBar
from ml_tools.utils.parallel import ray_context

def plot_ref_vs_pred(models:                  Dict[str, PredictionStrategy],
                     series_collection:       SeriesCollection,
                     fig_name:                str = 'ref_vs_pred',
                     state_index:             int = -1,
                     array_index:             int = 0,
                     error_bands:             List[float] = [5.0, 10.0],
                     title:                   bool = True,
                     predicted_feature_label: Optional[str] = None) -> None:
    """ Function for plotting reference vs. predicted results of a collection of models

    Parameters
    ----------
    models : Dict[str, PredictionStrategy]
        The collection of models (i.e. prediction strategies) whose predictions are to be plotted.
        The dictionary key will be the label used for the model in the plot legend
    series_collection : SeriesCollection
        The collection of state series to use for plotting
    state_index : int
        The index of the state in the series to be plotted (Default: -1)
    array_index : int
        The index of the predicted value array to be plotted (Default: 0)
    fig_name : str
        A name for the figure that is generated (Default: 'ref_vs_pred')
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
                          for series in series_collection])

    # Store legend handles for model data
    legend_handles = []
    legend_labels = []

    for label, model in models.items():
        assert model.isTrained
        assert model.predicted_feature == predicted_feature
        predicted = np.asarray([series[state_index][array_index] for series in model.predict(series_collection)])
        # Plot with alpha for visualization but create separate legend handle
        line = plt.plot(reference, predicted, '.', alpha=0.1, markersize=4)[0]
        # Create legend handle without alpha
        legend_handle = Line2D([0], [0], marker='o', color=line.get_color(),
                              linestyle='None', markersize=8, label=label)
        legend_handles.append(legend_handle)
        legend_labels.append(label)

    max_val = max(*plt.xlim(), *plt.ylim())
    plt.axis([0, max_val, 0, max_val])

    plt.plot([0, max_val], [0, max_val], '--k', label='Reference')
    grays = np.linspace(0.3, 0.7, len(error_bands))
    for gray, band in zip(grays, sorted(error_bands)):
        percent = band / 100.0
        x = np.linspace(0, max_val, 100)
        plt.plot(x, (1 + percent) * x, '--', color=(gray, gray, gray), label=f'+{band:.1f}%')
        plt.plot(x, (1 - percent) * x, '--', color=(gray, gray, gray), label=f'-{band:.1f}%')

    plt.grid(True)
    predicted_feature_label = predicted_feature if predicted_feature_label is None else predicted_feature_label
    plt.xlabel('Reference ' + predicted_feature_label, fontsize=14)
    plt.ylabel('Predicted ' + predicted_feature_label, fontsize=14)
    if title:
        plt.title('Reference vs. Predicted ' + predicted_feature_label, fontsize=16)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Create legend with custom handles for model data and automatic handles for other elements
    reference_line = Line2D([0], [0], color='black', linestyle='--', label='Reference')
    all_handles = legend_handles + [reference_line]
    all_labels = legend_labels + ['Reference']

    # Add error band handles
    for gray, band in zip(grays, sorted(error_bands)):
        error_line = Line2D([0], [0], color=(gray, gray, gray), linestyle='--', label=f'±{band:.1f}%')
        all_handles.append(error_line)
        all_labels.append(f'±{band:.1f}%')

    plt.legend(handles=all_handles, labels=all_labels, fontsize=12)
    plt.savefig(fig_name+'.png', dpi=600, bbox_inches='tight')
    plt.close()




def plot_hist(models:                  Dict[str, PredictionStrategy],
              series_collection:       SeriesCollection,
              fig_name:                str = 'hist',
              state_index:             int = -1,
              array_index:             int = 0,
              predicted_feature_label: Optional[str] = None) -> None:
    """ Function for plotting reference minus predicted histograms of a collection of models

    Parameters
    ----------
    models : Dict[str, PredictionStrategy]
        The collection of models (i.e. prediction strategies) whose predictions are to be plotted.
        The dictionary key will be the label used for the model in the plot legend
    series_collection : SeriesCollection
        The state series to use for plotting
    state_index : int
        The index of the state in the series to be plotted (Default: -1)
    array_index : int
        The index of the predicted value array to be plotted (Default: 0)
    fig_name : str
        A name for the figure that is generated (Default: 'hist')
    predicted_feature_label : Optional[str]
        The label to use for the predicted feature (Default: predicted feature label from models)
    """

    diffs             = []
    maxdiff           = 0.
    predicted_feature = next(iter(models.values())).predicted_feature
    reference         = np.asarray([series[state_index][predicted_feature][array_index]
                            for series in series_collection])
    for label, model in models.items():
        assert model.isTrained
        assert model.predicted_feature == predicted_feature
        predicted = np.asarray([series[state_index][array_index] for series in model.predict(series_collection)])
        diffs.append(reference - predicted)
        maxdiff = max(maxdiff, np.max(np.abs(diffs[-1])))

    colors = plt.cm.get_cmap('tab10').colors
    bins = np.linspace(-maxdiff, maxdiff, 100, endpoint=True)
    for i, (label, diff) in enumerate(zip(models.keys(), diffs)):
        plt.hist(diff, bins, histtype='step', linewidth=1.5, label=label, color=colors[i % len(colors)])

    plt.grid(True)
    predicted_feature_label = predicted_feature if predicted_feature_label is None else predicted_feature_label
    plt.xlabel('Reference - Predicted ' + predicted_feature_label)
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(fig_name+'.png')
    plt.close()


def plot_sensitivities(models:                  Dict[str, PredictionStrategy],
                       series_collection:       SeriesCollection,
                       perturbators:            Dict[str, FeaturePerturbator],
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
    series_collection : SeriesCollection
        The collection of state series to use for evaluating the sensitivities
    perturbators : Dict[str, FeaturePerturbator]
        The perturbators to use for perturbing the state features
    number_of_perturbations : int
        The number of perturbation realizations to perform
    state_index : int
        The index of the state in the series to be plotted (Default: -1)
    array_index : int
        The index of the predicted value array to be plotted (Default: 0)
    fig_name_prefix : str
        The prefix for the figure files that will be created (Default: 'box_plot')
    num_procs : int
        The number of parallel processors to use when perturbing states
    """

    perturbations = []
    for _ in range(number_of_perturbations):
        perturbations.append([])
        for series in series_collection:
            perturbations[-1].append(State.perturb_states(perturbators, series, num_procs))

    predicted_feature = next(iter(models.values())).predicted_feature

    for label, model in models.items():
        assert model.isTrained
        assert model.predicted_feature == predicted_feature

        predicted = np.asarray([series[state_index][array_index] for series in model.predict(series_collection)])
        results   = [[] for series in series_collection]
        for perturbation in perturbations:
            perturbed_predicted = model.predict(perturbation)
            for i in range(len(series_collection)):
                results[i].append(perturbed_predicted[i][state_index][array_index])

        order = np.argsort(predicted)
        plt.figure(figsize=(12, 5))
        plt.boxplot([results[i] for i in order])
        plt.plot(range(1,len(order)+1), [predicted[i] for i in order],'.r')
        ticks = plt.gca().get_xticks()
        plt.xticks(ticks, [str(int(tick)) if int(tick) % 10 == 0 else '' for tick in ticks])
        plt.savefig(fig_name_prefix+"_"+label+'.png', dpi=600, bbox_inches='tight')
        plt.close()



def print_metrics(models:            Dict[str, PredictionStrategy],
                  series_collection: SeriesCollection,
                  state_index:       int = -1,
                  array_index:       int = 0,
                  output_file:       Optional[str] = None) -> None:
    """ Function for printing the statistical metrics (average, std.dev., rms, max) of models to the screen

    Parameters
    ----------
    models : Dict[str, PredictionStrategy]
        The collection of models (i.e. prediction strategies) whose predictions are to be plotted.
        The dictionary key will be the label used for the model in the plot legend
    series_collection : SeriesCollection
        The state series to use for plotting
    state_index : int
        The index of the state in the series to be plotted (Default: -1)
    array_index : int
        The index of the predicted value array to be plotted (Default: 0)
    output_file : Optional[str]
        An optional file in which to output
    """

    predicted_feature = next(iter(models.values())).predicted_feature
    reference         = np.asarray([series[state_index][predicted_feature][array_index]
                            for series in series_collection])

    padding = np.max([len(label) for label in models.keys()]) + 3
    fmtstr  = f'{{0:{padding}s}} {{1:8.5f}} {{2:7.5f}} {{3:7.5f}} {{4:7.5f}} {{5:9.3e}}'
    header  = ' ' * padding + '   Avg     Std     RMS     Max   Time/State'
    print(header)
    if output_file:
        with open(output_file, 'a') as f:
            f.write(f"{header}\n")
    for label, model in models.items():
        start = time.time()
        predicted = np.asarray([series[state_index][array_index] for series in model.predict(series_collection)])
        dt = time.time() - start
        diff = reference - predicted
        rms = np.sqrt(np.dot(diff.flatten(),diff.flatten()))/float(len(diff))
        maxdiff = np.max(np.abs(diff))
        avg = np.mean(diff)
        std = np.std(diff)
        metrics = fmtstr.format(label + ' :', avg, std, rms, maxdiff, dt/len(series_collection))
        print(metrics)
        if output_file:
            with open(output_file, 'a') as f:
                f.write(f"{metrics}\n")



def plot_corr_matrix(input_features:    List[str],
                     series_collection: SeriesCollection,
                     state_index:       int = -1,
                     fig_name:          str = 'corr_matrix') -> None:
    """ Function for plotting the correlation matrix of a given set of input features

    Parameters
    ----------
    input_features : List[str]
        A list specifying the input features whose correlations are to be plotted.
    series_collection : SeriesCollection
        The state series to use for plotting
    state_index : int
        The index of the state in the series to be plotted (Default: -1)
    fig_name : str
        A name for the figure that is generated (Default: 'corr_matrix')
    """

    series_collection = SeriesCollection(series_collection)
    X = series_collection.to_dataframe(input_features)

    if state_index < 0:
        max_index = X.index.get_level_values('state_index').max()
        state_index = max_index + 1 + state_index

    valid_indices = X.index.get_level_values('state_index').unique()
    assert state_index in valid_indices, \
        f"state_index {state_index} is out of bounds (valid: {valid_indices.min()} to {valid_indices.max()})"

    input_features = [col for col in X.columns if any(col == prefix or col.startswith(prefix) for prefix in input_features)]
    num_features   = len(input_features)

    X            = X.loc[(slice(None), state_index), :]
    X            = X[input_features]
    corr_matrix  = X.corr().fillna(0)
    mask         = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(8, 6))
    annotate = num_features <= 12
    sns.heatmap(
        corr_matrix,
        mask        = mask,
        annot       = annotate,
        xticklabels = input_features,
        yticklabels = input_features,
        cmap        = 'coolwarm',
        fmt         = '.2f' if annotate else "",
        cbar        = True)

    label_font_scale = max(0.3, min(1.2, 10 / num_features))
    valid_rows       = sorted(set(np.where(~mask)[0]))
    valid_cols       = sorted(set(np.where(~mask)[1]))
    row_ticks        = [i + 0.5 for i in valid_rows]
    col_ticks        = [i + 0.5 for i in valid_cols]
    tick_positions   = np.arange(num_features)
    plt.xticks(ticks=col_ticks, labels=[input_features[i] for i in valid_cols],
               rotation=45, ha='right', fontsize=10 * label_font_scale)
    plt.yticks(ticks=row_ticks, labels=[input_features[i] for i in valid_rows],
               rotation=0, fontsize=10 * label_font_scale)
    plt.tight_layout()
    plt.savefig(fig_name+'.png', dpi=600, bbox_inches='tight')
    plt.close()


def plot_ice_pdp(models:                  Dict[str, PredictionStrategy],
                 series_collection:       SeriesCollection,
                 input_feature:           str,
                 fig_name_prefix:         str = 'ice_pdp',
                 state_index:             int = -1,
                 input_index:             int = 0,
                 output_index:            int = 0,
                 num_points:              int = 50,
                 silent:                  bool = False,
                 num_procs:               int = 1,
                 predicted_feature_label: Optional[str] = None) -> None:
    """ Function to plot ICE/PDP feature analyses for a given set of models.

    Parameters
    ----------
    models : Dict[str, PredictionStrategy]
        The collection of models (i.e. prediction strategies) whose predictions are to be plotted.
        The dictionary key will be the label used for the model in the plot legend
    series_collection : SeriesCollection
        The state series to use for plotting
    input_feature : str
        The input feature to generate ICE / PDP plots for
    fig_name_prefix : str
        The prefix for the figure files that will be created (Default: 'ice_pdp')
    state_index : int
        The index of the state in the series to be analyzed (Default: -1, last state)
    input_index : int
        The index of the input feature value array to be plotted (Default: 0)
    output_index : int
        The index of the predicted value array to be plotted (Default: 0)
    num_points : int
        The number of sampled points for ICE and PDP curves (Default: 50)
    predicted_feature_label : Optional[str]
        The label to use for the predicted feature (Default: predicted feature label from models)
    silent : bool
        A flag indicating whether or not to display the progress bar to the screen
    num_procs : int
        The number of parallel processors to use (Default: 1)
    """

    predicted_feature = next(iter(models.values())).predicted_feature

    values = np.asarray([series[state_index][input_feature][input_index] for series in series_collection])
    values = np.linspace(np.min(values), np.max(values), num_points)

    chunk_size = max(1, len(values) // num_procs)
    batches    = [values[i:i + chunk_size] for i in range(0, len(values), chunk_size)]

    with ray_context(num_cpus=num_procs):
        for label, model in models.items():
            assert model.isTrained, f"Model {label} must be trained before ICE/PDP analysis."

            if not silent:
                print(f"Generating ICE/PDP plot: {fig_name_prefix}_{label}_{input_feature}.png")
                statusbar = StatusBar(len(values) * len(series_collection))

            args = [(model, batch, series_collection, input_feature, state_index, input_index, output_index)
                    for batch in batches]


            jobs       = [_process_ice_pdp_batch.remote(*arg) for arg in args]
            unfinished = list(jobs)
            ice_data   = []
            completed  = 0

            while unfinished:
                ready, unfinished = ray.wait(unfinished, num_returns=1)
                result = ray.get(ready[0])
                ice_data.extend(result)
                completed += len(result)
                if not silent:
                    statusbar.update(completed)

            if not silent:
                statusbar.finalize()

            ice_df = pd.DataFrame(ice_data)

            plt.figure(figsize=(10, 6))
            sns.lineplot(data=ice_df, x='value', y='prediction', hue='sample',
                         estimator=None, alpha=0.3, legend=False, linewidth=1)

            pdp_df = ice_df.groupby('value')['prediction'].mean().reset_index()
            sns.lineplot(data=pdp_df, x='value', y='prediction', color='red', label='PDP', linewidth=2)

            sns.rugplot(x=values, height=0.03, color='black', alpha=0.3)

            predicted_feature_label = predicted_feature_label or predicted_feature
            plt.xlabel(input_feature)
            plt.ylabel(predicted_feature_label)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{fig_name_prefix}_{label}_{input_feature}.png", dpi=600, bbox_inches='tight')
            plt.close()

@ray.remote
def _process_ice_pdp_batch(model:              PredictionStrategy,
                           batch:              np.ndarray,
                           series_collection:  SeriesCollection,
                           input_feature:      str,
                           state_index:        int,
                           input_index:        int,
                           output_index:       int) -> List[Dict[str, Any]]:
    """ Private helper function used by `plot_ice_pdp` for parallel batch processing.

    This function is not intended to be called directly. For an understanding of the parameters
    parameter, please refer to `plot_ice_pdp`.
    """

    batch_results = []
    for value in batch:
        series_collection_perturbed = deepcopy(series_collection)
        for series in series_collection_perturbed:
            series[state_index][input_feature][input_index] = value

        predictions = model.predict(series_collection_perturbed)

        batch_results.extend([
            {'sample': i, 'value': value, 'prediction': series[state_index][output_index]}
            for i, series in enumerate(predictions)
        ])
    return batch_results


def plot_shap(models:            Dict[str, PredictionStrategy],
              series_collection: SeriesCollection,
              feature_plots:     Dict[str, List[str]],
              algorithm:         str = 'auto',
              fig_name_prefix:   str = 'shap',
              state_index:       int = -1,
              array_index:       int = 0,
              num_samples:       int = 50,
              silent:            bool = False,
              num_procs:         int = 1) -> None:
    """ Function to plot SHAP feature importance summary for a given set of models.

    Parameters
    ----------
    models : Dict[str, PredictionStrategy]
        The collection of models (i.e. prediction strategies) whose sensitivities are to be plotted.
        The dictionary key will be the suffix of the figure file name
    series_collection : SeriesCollection
        The state series to use for plotting
    feature_plots : Dict[str, List[str]]
        Specification for what sets of input features to make SHAP plots for. The key corresponds to the
        set label for file naming, and the value the list of input features the corresponding plot will include
    algorithm : str
        The SHAP algorithm to use, either 'auto', 'permutation', or 'partition' (Default: 'auto')
    fig_name_prefix : str
        The prefix for the figure files that will be created (Default: 'shap')
    state_index : int
        The index of the state in the series to be analyzed (Default: -1, last state).
    array_index : int
        The index of the predicted value array to be plotted (Default: 0)
    num_samples : int
        The number of sampled points for assessing the SHAP values. (Default: 50)
    silent : bool
        A flag indicating whether or not to display the progress bar to the screen
    num_procs : int
        The number of parallel processors to use when performing the SHAP evaluation (Default: 1)
    """

    series_collection  = series_collection.random_sample(min(num_samples, len(series_collection)))
    df                 = series_collection.to_dataframe()
    all_input_features = list(df.columns)

    min_samples_per_batch = 5 # This is needed to allow SHAP to correctly perform clustering
    max_num_batches       = max(1, len(df) // min_samples_per_batch)
    num_procs             = min(num_procs, max_num_batches, os.cpu_count() or 1)

    with ray_context(num_cpus=num_procs):
        for model_label, model in models.items():
            assert model.isTrained, f"Model {model_label} must be trained before SHAP analysis."

            processed_inputs = model.preprocess_inputs(series_collection)
            batches          = np.array_split(processed_inputs, num_procs)
            batches          = [b for b in batches if len(b) > 0]

            if not silent:
                print(f"Generating SHAP plot: {fig_name_prefix}_{model_label}.png")
                statusbar = StatusBar(sum(len(b) for b in batches))

            args = [(model, batch, state_index, array_index, algorithm) for batch in batches]

            jobs        = [_process_shap_batch.remote(*arg) for arg in args]
            unfinished  = list(jobs)
            shap_chunks = []
            completed   = 0

            while unfinished:
                ready, unfinished = ray.wait(unfinished, num_returns=1)
                result = ray.get(ready[0])
                shap_chunks.append(result)
                completed += len(result.values)
                if not silent:
                    statusbar.update(completed)

            shap_values    = np.concatenate([chunk.values for chunk in shap_chunks], axis=0)
            feature_values = np.concatenate([chunk.data   for chunk in shap_chunks], axis=0)

            # Postprocess feature values back to original scale
            for f, processor in model.input_features.items():
                feature_indices = [i for i, feature in enumerate(all_input_features)
                                   if feature == f or feature.startswith(f)]
                feature_values[:, feature_indices] = processor.postprocess(feature_values[:, feature_indices])

            # Plot SHAP summary plots for each specified feature set
            for feature_set, features in feature_plots.items():
                selected_input_feature_indices = [i for i, feature in enumerate(all_input_features)
                                                  if any(feature == f or feature.startswith(f) for f in features)]
                selected_feature_names         = [all_input_features[i] for i in selected_input_feature_indices]
                selected_shap_values           = shap_values[:, selected_input_feature_indices]
                selected_feature_values        = feature_values[:, selected_input_feature_indices]

                plt.figure(figsize=(10, 6))
                shap.summary_plot(selected_shap_values,
                                  feature_names = selected_feature_names,
                                  features      = selected_feature_values,
                                  show          = False)

                plt.savefig(fig_name_prefix+"_"+feature_set+"_"+model_label+'.png', dpi=600, bbox_inches='tight')
                plt.close()

@ray.remote
def _process_shap_batch(model:       PredictionStrategy,
                        batch:       np.ndarray,
                        state_index: int,
                        array_index: int,
                        algorithm:   str) -> shap.Explanation:
    """ Private helper function used by `plot_shap` for parallel batch processing.

    This function is not intended to be called directly. For a description of the parameters,
    refer to the documentation of `plot_shap`.
    """

    state_index = state_index if state_index >= 0 else batch.shape[1] + state_index

    def shap_wrapper(shap_inputs: np.ndarray) -> np.ndarray:
        """ Wrapper function which is necessary due to shap.Explainer requiring np.ndarray inputs
        """
        full = np.repeat(batch, shap_inputs.shape[0], axis=0)
        full[:, state_index, :] = shap_inputs
        padded = model.predict_processed_inputs(full)
        predictions = model.post_process_outputs(padded)
        return np.asarray([series[state_index][array_index] for series in predictions])

    shap_batch = batch[:, state_index, :]
    explainer = shap.Explainer(shap_wrapper, shap_batch, algorithm=algorithm)
    return explainer(shap_batch)
