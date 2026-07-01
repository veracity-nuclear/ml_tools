from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import pylab as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from ml_tools.model.feature_perturbator import FeaturePerturbator
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.model.state import State, SeriesCollection


ValueTransform = Callable[[np.ndarray], np.ndarray]
PerturbatorMap = Mapping[str, FeaturePerturbator]
PerturbatorInput = Mapping[str, FeaturePerturbator | PerturbatorMap]


class PredictionResults:
    """Container for extracted model prediction results.

    This class defines the model outputs that should be extracted from one or
    more :class:`~ml_tools.model.state.SeriesCollection` objects for downstream
    evaluation and plotting. Each :class:`Spec` identifies one trained model,
    reference series collection, output feature, state index, and array index
    to extract as one column in the result arrays.

    Parameters
    ----------
    specs : Sequence[PredictionResults.Spec]
        Specifications describing the model output comparisons to extract.

    Attributes
    ----------
    specs : list[PredictionResults.Spec]
        Stored copy of the prediction result specifications.
    reference_values : numpy.ndarray
        Extracted reference values with shape ``(num_series, num_specs)``. Each
        column corresponds to one entry in ``specs``.
    predicted_values : numpy.ndarray
        Extracted predicted values with shape ``(num_series, num_specs)``. Each
        column corresponds to one entry in ``specs``.
    labels : list[str]
        Column labels corresponding to ``reference_values`` and
        ``predicted_values``.
    """

    @dataclass(frozen=True, kw_only=True)
    class Spec:
        """Specification for one model output comparison.

        Parameters
        ----------
        label : str
            Label used to identify this model/output comparison in result
            tables, metrics, and plots.
        model : PredictionStrategy
            Trained prediction strategy used to generate predicted output
            values.
        series_collection : SeriesCollection
            Reference state series collection used as model input and as the
            source of reference output values for this comparison.
        predicted_feature : str
            Name of the predicted feature to extract from the model predictions
            and reference series collection.
        state_index : int, optional
            State index to extract from each series. Negative values follow
            normal Python indexing. Default is -1, which extracts the last state
            in each series.
        array_index : int, optional
            Index within the selected predicted feature array to extract.
        value_transform : ValueTransform, optional
            Optional callable applied to the extracted reference and predicted
            value arrays. The callable must accept one ``numpy.ndarray`` and
            return one ``numpy.ndarray`` with compatible shape.
        """

        label:             str
        model:             PredictionStrategy
        series_collection: SeriesCollection
        predicted_feature: str
        state_index:       int = -1
        array_index:       int = 0
        value_transform:   Optional[ValueTransform] = None

    def __init__(self, specs: Sequence[Spec]) -> None:
        self.specs = list(specs)
        assert len(self.specs) > 0, "PredictionResults requires at least one spec."
        self.labels = [self._get_label(spec) for spec in self.specs]
        self._check_unique_labels(self.labels)
        self.reference_values, self.predicted_values = self._extract_values()


    def to_dataframe(self) -> pd.DataFrame:
        """Return extracted prediction results as a wide dataframe.

        The returned dataframe contains one row per series and paired reference
        and predicted columns for each :class:`Spec`. Column names are prefixed
        with ``"reference:"`` and ``"predicted:"`` followed by the spec label.

        Returns
        -------
        pandas.DataFrame
            Wide dataframe containing a ``series_index`` column followed by one
            reference column and one predicted column for each spec.
        """
        data = {}
        for index, label in enumerate(self.labels):
            data[f"reference:{label}"] = self.reference_values[:, index]
            data[f"predicted:{label}"] = self.predicted_values[:, index]
        return pd.DataFrame(data).rename_axis("series_index").reset_index()


    def plot_ref_vs_pred(self,
                         fig_name:    str = "ref_vs_pred",
                         error_bands: Optional[Sequence[float]] = None,
                         title:       bool = True,
                         value_label: Optional[str] = None,
                         alpha:       float = 0.1,
                         markersize:  float = 4.0) -> None:
        """Plot reference values against predicted values.

        This method plots each extracted spec as a scatter series using the
        materialized ``reference_values`` and ``predicted_values`` arrays. Model
        predictions are not recomputed. A one-to-one reference line and optional
        percentage error bands are drawn using the same units as the extracted
        values, after any spec-level ``value_transform`` has already been
        applied.

        Parameters
        ----------
        fig_name : str, optional
            Output figure path without the ``.png`` suffix. Default is
            ``"ref_vs_pred"``.
        error_bands : sequence of float, optional
            Percentage error bands to draw around the one-to-one reference
            line. For example, ``5.0`` draws ``+5%`` and ``-5%`` bands. If
            omitted, ``[5.0, 10.0]`` is used.
        title : bool, optional
            Whether to include a plot title. Default is ``True``.
        value_label : str, optional
            Axis label for the compared value. If omitted, the label is derived
            from the shared ``predicted_feature`` and ``array_index`` when all
            specs reference the same output component.
        alpha : float, optional
            Scatter marker opacity for plotted prediction points. Default is
            ``0.1``.
        markersize : float, optional
            Scatter marker size. Default is ``4.0``.

        Returns
        -------
        None
            The plot is saved to ``fig_name + ".png"`` and the matplotlib
            figure is closed.

        Raises
        ------
        AssertionError
            If axis limits cannot be determined from finite data.
        """
        error_bands = [5.0, 10.0] if error_bands is None else list(error_bands)
        value_label = value_label or self._get_value_label()

        plt.figure(figsize=(10, 6))
        legend_handles = []
        legend_labels = []

        for index, spec in enumerate(self.specs):
            line = plt.plot(self.reference_values[:, index], self.predicted_values[:, index], ".",
                            alpha=alpha, markersize=markersize)[0]
            legend_handles.append(Line2D([0], [0], marker="o", color=line.get_color(),
                                         linestyle="None", markersize=8, label=spec.label))
            legend_labels.append(spec.label)

        min_val, max_val = self._get_ref_vs_pred_limits()
        plt.axis([min_val, max_val, min_val, max_val])

        x = np.linspace(min_val, max_val, 100)
        plt.plot(x, x, "--k", label="Reference")

        grays = np.linspace(0.3, 0.7, len(error_bands)) if error_bands else []
        for gray, band in zip(grays, sorted(error_bands)):
            percent = band / 100.0
            color = (gray, gray, gray)
            plt.plot(x, (1 + percent) * x, "--", color=color)
            plt.plot(x, (1 - percent) * x, "--", color=color)

        plt.grid(True)
        plt.xlabel("Reference " + value_label, fontsize=14)
        plt.ylabel("Predicted " + value_label, fontsize=14)
        if title:
            plt.title("Reference vs. Predicted " + value_label, fontsize=16)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        all_handles = legend_handles + [Line2D([0], [0], color="black", linestyle="--", label="Reference")]
        all_labels = legend_labels + ["Reference"]

        for gray, band in zip(grays, sorted(error_bands)):
            all_handles.append(Line2D([0], [0], color=(gray, gray, gray), linestyle="--", label=f"+/-{band:.1f}%"))
            all_labels.append(f"+/-{band:.1f}%")

        plt.legend(handles=all_handles, labels=all_labels, fontsize=12)
        plt.savefig(fig_name + ".png", dpi=600, bbox_inches="tight")
        plt.close()


    def plot_hist(self,
                  fig_name:    str = "hist",
                  bins:        int = 100,
                  value_label: Optional[str] = None,
                  linewidth:   float = 1.5) -> None:
        """Plot histograms of reference-minus-predicted residuals.

        This method computes residuals from the materialized
        ``reference_values`` and ``predicted_values`` arrays. Model predictions
        are not recomputed. Each spec is plotted as one step histogram using a
        shared symmetric bin range centered on zero.

        Parameters
        ----------
        fig_name : str, optional
            Output figure path without the ``.png`` suffix. Default is
            ``"hist"``.
        bins : int, optional
            Number of histogram bins. Default is ``100``.
        value_label : str, optional
            Axis label for the compared value. If omitted, the label is derived
            from the shared ``predicted_feature`` and ``array_index`` when all
            specs reference the same output component.
        linewidth : float, optional
            Histogram step line width. Default is ``1.5``.

        Returns
        -------
        None
            The plot is saved to ``fig_name + ".png"`` and the matplotlib
            figure is closed.

        Raises
        ------
        AssertionError
            If no finite residuals are available.
        """
        value_label = value_label or self._get_value_label()
        residuals   = self.reference_values - self.predicted_values
        max_diff    = self._get_max_abs_finite_value(residuals)
        bins        = np.linspace(-max_diff, max_diff, bins, endpoint=True)
        colors      = plt.get_cmap("tab10").colors

        plt.figure()
        for index, label in enumerate(self.labels):
            plt.hist(residuals[:, index], bins, histtype="step", linewidth=linewidth, label=label,
                     color=colors[index % len(colors)])

        plt.grid(True)
        plt.xlabel("Reference - Predicted " + value_label)
        plt.ylabel("Count")
        plt.legend()
        plt.savefig(fig_name + ".png")
        plt.close()


    def print_metrics(self, output_file: Optional[str] = None) -> None:
        """Print residual summary metrics for each spec.

        Metrics are computed from the materialized residuals
        ``reference_values - predicted_values``. Model predictions are not
        recomputed. The printed columns are the average residual, residual
        standard deviation, root-mean-square residual, and maximum absolute
        residual.

        Parameters
        ----------
        output_file : str, optional
            Optional file path to append the metrics table to. If omitted,
            metrics are printed only to standard output.

        Returns
        -------
        None
            Metrics are printed and optionally appended to ``output_file``.

        """
        def print_or_write_line(line: str,
                                output_file: Optional[str]) -> None:
            """Print a line and optionally append it to a file."""
            print(line)
            if output_file:
                with open(output_file, "a", encoding="utf-8") as file:
                    file.write(f"{line}\n")

        padding = max(len(label) for label in self.labels) + 3
        fmtstr = f"{{0:{padding}s}} {{1:8.5f}} {{2:7.5f}} {{3:7.5f}} {{4:7.5f}}"
        header = " " * padding + "   Avg     Std     RMS     Max"

        print_or_write_line(header, output_file)
        residuals = self.reference_values - self.predicted_values
        for index, label in enumerate(self.labels):
            diff = residuals[:, index]
            metrics = fmtstr.format(label + " :",
                                    np.mean(diff),
                                    np.std(diff),
                                    np.sqrt(np.mean(diff * diff)),
                                    np.max(np.abs(diff)))
            print_or_write_line(metrics, output_file)


    def plot_sensitivities(self,
                           perturbators:            PerturbatorInput,
                           number_of_perturbations: int,
                           fig_name_prefix:         str = "box_plot",
                           num_procs:               int = 1) -> None:
        """Plot prediction sensitivities from perturbed state series.

        For each spec, this method perturbs the spec's
        :class:`~ml_tools.model.state.SeriesCollection`, reruns the spec's
        model on each perturbed realization, extracts the same output component
        represented by the spec, and plots the perturbed prediction distribution
        as a boxplot. The unperturbed predictions are overlaid as red points and
        samples are ordered by those unperturbed predictions.

        ``perturbators`` may be provided either as a shared feature perturbator
        map used for every spec, or as a per-label mapping:

        ``{"feature": perturbator}``
            Shared perturbators for all specs.

        ``{"spec label": {"feature": perturbator}}``
            Separate perturbators for each spec label.

        Parameters
        ----------
        perturbators : PerturbatorInput
            Shared or per-label mapping of input feature names to
            :class:`~ml_tools.model.feature_perturbator.FeaturePerturbator`
            instances. Perturbed feature names must be present in the
            corresponding spec model's ``input_features``.
        number_of_perturbations : int
            Number of perturbed realizations to generate for each spec.
        fig_name_prefix : str, optional
            Prefix for generated figure names. Each figure is saved as
            ``fig_name_prefix + "_" + spec.label + ".png"``. Default is
            ``"box_plot"``.
        num_procs : int, optional
            Number of parallel processors to use when perturbing states and
            running model predictions. Default is ``1``.

        Returns
        -------
        None
            One sensitivity plot is saved for each spec.

        Raises
        ------
        AssertionError
            If ``number_of_perturbations`` is not positive, ``num_procs`` is
            not positive, or perturbator validation fails.
        """
        assert number_of_perturbations > 0, "number_of_perturbations must be positive."
        assert num_procs > 0, "num_procs must be positive."

        perturbators_by_label = self._normalize_perturbators(perturbators)

        for index, spec in enumerate(self.specs):
            predicted = self.predicted_values[:, index]
            perturbed_results = [[] for _ in spec.series_collection]

            for _ in range(number_of_perturbations):
                perturbed_collection = SeriesCollection([
                    State.perturb_states(perturbators_by_label[spec.label], series, num_procs=num_procs)
                    for series in spec.series_collection
                ])
                perturbed_predictions = spec.model.predict(perturbed_collection, num_procs=num_procs)
                perturbed_values = self._extract_spec_values(spec, perturbed_predictions)

                for series_index, value in enumerate(perturbed_values):
                    perturbed_results[series_index].append(value)

            order = np.argsort(predicted)
            plt.figure(figsize=(12, 5))
            plt.boxplot([perturbed_results[i] for i in order])
            plt.plot(range(1, len(order) + 1), [predicted[i] for i in order], ".r")
            ticks = plt.gca().get_xticks()
            plt.xticks(ticks, [str(int(tick)) if int(tick) % 10 == 0 else "" for tick in ticks])
            plt.savefig(f"{fig_name_prefix}_{spec.label}.png", dpi=600, bbox_inches="tight")
            plt.close()


    def _extract_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract all spec values into reference and predicted arrays."""
        reference_columns = []
        predicted_columns = []
        predictions = {}

        for spec in self.specs:
            assert spec.model.isTrained, f"Model '{spec.label}' must be trained before extracting results."
            assert spec.predicted_feature in spec.model.predicted_features, \
                f"Model '{spec.label}' does not predict '{spec.predicted_feature}'."

            prediction_key = (id(spec.model), id(spec.series_collection))
            if prediction_key not in predictions:
                predictions[prediction_key] = spec.model.predict(spec.series_collection)

            reference = self._extract_spec_values(spec, spec.series_collection)
            predicted = self._extract_spec_values(spec, predictions[prediction_key])
            self._check_matching_shape(reference, predicted, spec)

            reference_columns.append(reference)
            predicted_columns.append(predicted)

        if not reference_columns:
            return np.empty((0, 0)), np.empty((0, 0))

        self._check_matching_column_lengths(reference_columns)
        return np.column_stack(reference_columns), np.column_stack(predicted_columns)


    @staticmethod
    def _extract_spec_values(spec: Spec,
                             series_collection: SeriesCollection) -> np.ndarray:
        """Extract values for one spec from a series collection."""
        values = np.asarray([series[spec.state_index][spec.predicted_feature][spec.array_index]
                             for series in series_collection])
        if spec.value_transform is None:
            return values

        transformed = np.asarray(spec.value_transform(values))
        assert transformed.shape == values.shape, \
            f"Value transform for '{spec.label}' changed shape from {values.shape} to {transformed.shape}."
        return transformed


    @staticmethod
    def _get_label(spec: Spec) -> str:
        """Return the dataframe column label for one spec."""
        return spec.label


    def _get_value_label(self) -> str:
        """Return a default value label for plot axes."""
        feature_indices = {(spec.predicted_feature, spec.array_index) for spec in self.specs}
        if len(feature_indices) == 1:
            predicted_feature, array_index = next(iter(feature_indices))
            return f"{predicted_feature}[{array_index}]"

        return "Value"


    def _get_ref_vs_pred_limits(self) -> tuple[float, float]:
        """Return equal x/y axis limits for reference-vs-predicted plots."""
        values = np.concatenate([self.reference_values.reshape(-1),
                                 self.predicted_values.reshape(-1)])
        finite_values = values[np.isfinite(values)]
        assert finite_values.size > 0, "Cannot plot reference vs. predicted values without finite data."

        min_val = float(np.min(finite_values))
        max_val = float(np.max(finite_values))
        if np.isclose(min_val, max_val):
            padding = 1.0 if np.isclose(max_val, 0.0) else 0.05 * abs(max_val)
            min_val -= padding
            max_val += padding
        else:
            padding = 0.02 * (max_val - min_val)
            min_val -= padding
            max_val += padding

        return min(min_val, 0.0), max(max_val, 0.0)


    @staticmethod
    def _get_max_abs_finite_value(values: np.ndarray) -> float:
        """Return a positive finite limit for symmetric residual plots."""
        finite_values = values[np.isfinite(values)]
        assert finite_values.size > 0, "Cannot plot residual histogram without finite data."

        max_value = float(np.max(np.abs(finite_values)))
        if np.isclose(max_value, 0.0):
            return 1.0
        return max_value


    def _normalize_perturbators(self, perturbators: PerturbatorInput) -> dict[str, dict[str, FeaturePerturbator]]:
        """Normalize shared or per-label perturbators into a per-label mapping."""
        assert len(perturbators) > 0, "perturbators must be non-empty."

        if all(isinstance(value, FeaturePerturbator) for value in perturbators.values()):
            normalized = {spec.label: dict(perturbators) for spec in self.specs}
        else:
            assert all(isinstance(value, Mapping) for value in perturbators.values()), \
                "perturbators must be either a feature->perturbator map or a label->feature->perturbator map."
            missing_labels = sorted(set(self.labels) - set(perturbators))
            unknown_labels = sorted(set(perturbators) - set(self.labels))
            assert not missing_labels, f"Missing perturbators for labels: {missing_labels}."
            assert not unknown_labels, f"Unknown perturbator labels: {unknown_labels}."
            normalized = {label: dict(feature_perturbators)
                          for label, feature_perturbators in perturbators.items()}

        for spec in self.specs:
            feature_perturbators = normalized[spec.label]
            assert all(isinstance(value, FeaturePerturbator) for value in feature_perturbators.values()), \
                f"Perturbators for '{spec.label}' must all be FeaturePerturbator instances."
            missing_features = sorted(set(feature_perturbators) - set(spec.model.input_features))
            assert not missing_features, \
                f"Perturbed features for '{spec.label}' are not model input features: {missing_features}."

        return normalized


    @staticmethod
    def _check_matching_shape(reference: np.ndarray,
                              predicted: np.ndarray,
                              spec: Spec) -> None:
        """Check that one spec produced aligned reference and prediction arrays."""
        assert reference.shape == predicted.shape, \
            f"Spec '{spec.label}' reference shape {reference.shape} does not match predicted shape {predicted.shape}."


    @staticmethod
    def _check_matching_column_lengths(columns: Sequence[np.ndarray]) -> None:
        """Check that all specs can be stacked into a two-dimensional array."""
        expected_length = len(columns[0])
        for index, column in enumerate(columns[1:], start=1):
            assert len(column) == expected_length, \
                f"Spec at index {index} produced {len(column)} values; expected {expected_length}."


    @staticmethod
    def _check_unique_labels(labels: Sequence[str]) -> None:
        """Check that spec labels can be used as stable result identifiers."""
        seen = set()
        duplicates = []
        for label in labels:
            if label in seen and label not in duplicates:
                duplicates.append(label)
            seen.add(label)
        assert not duplicates, f"PredictionResults spec labels must be unique. Duplicates: {duplicates}."
