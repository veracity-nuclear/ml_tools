import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from ml_tools import SeriesCollection, State, StateSeries
from ml_tools.model.feature_processor import NoProcessing
from ml_tools.model.feature_perturbator import NonPerturbator
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.utils.prediction_results import PredictionResults


class DummyStrategy(PredictionStrategy):
    def __init__(self, multiplier: float = 2.0):
        super().__init__()
        self.input_features           = {"x": NoProcessing()}
        self.predicted_features       = {"y": NoProcessing()}
        self._predicted_feature_sizes = {"y": 1}
        self._trained                 = True
        self._multiplier              = multiplier

    @property
    def isTrained(self) -> bool:
        return self._trained

    def train(self, train_data, test_data=None, num_procs: int = 1) -> None:
        self._trained                 = True
        self._predicted_feature_sizes = {"y": 1}

    def _params_to_dict(self) -> dict:
        return {"multiplier": self._multiplier}

    @classmethod
    def read_from_file(cls, file_name: str):
        raise NotImplementedError

    def _predict_one(self, state_series: np.ndarray) -> np.ndarray:
        return state_series[:, :1] * self._multiplier


def test_prediction_results_extracts_reference_and_model_values():
    first_collection = SeriesCollection([
        StateSeries([State({"x": np.array([1.0]), "y": np.array([2.0])})]),
        StateSeries([State({"x": np.array([2.0]), "y": np.array([4.0])})]),
        StateSeries([State({"x": np.array([3.0]), "y": np.array([6.0])})]),
    ])
    second_collection = SeriesCollection([
        StateSeries([State({"x": np.array([4.0]), "y": np.array([8.0])})]),
        StateSeries([State({"x": np.array([5.0]), "y": np.array([10.0])})]),
        StateSeries([State({"x": np.array([6.0]), "y": np.array([12.0])})]),
    ])
    model = DummyStrategy()

    results = PredictionResults([
        PredictionResults.Spec(label="First",
                               model=model,
                               series_collection=first_collection,
                               predicted_feature="y"),
        PredictionResults.Spec(label="Second Scaled",
                               model=model,
                               series_collection=second_collection,
                               predicted_feature="y",
                               value_transform=lambda values: values / 2.0),
    ])

    np.testing.assert_allclose(results.reference_values,
                               np.asarray([[2.0, 4.0],
                                           [4.0, 5.0],
                                           [6.0, 6.0]]))
    np.testing.assert_allclose(results.predicted_values,
                               np.asarray([[2.0, 4.0],
                                           [4.0, 5.0],
                                           [6.0, 6.0]]))

    df = results.to_dataframe()
    assert list(df.columns) == [
        "series_index",
        "reference:First",
        "predicted:First",
        "reference:Second Scaled",
        "predicted:Second Scaled",
    ]
    np.testing.assert_allclose(df["reference:First"].to_numpy(), np.asarray([2.0, 4.0, 6.0]))


def test_prediction_results_plot_ref_vs_pred(tmp_path):
    series_collection = SeriesCollection([
        StateSeries([State({"x": np.array([1.0]), "y": np.array([2.0])})]),
        StateSeries([State({"x": np.array([2.0]), "y": np.array([4.0])})]),
        StateSeries([State({"x": np.array([3.0]), "y": np.array([6.0])})]),
    ])
    model = DummyStrategy()
    results = PredictionResults([
        PredictionResults.Spec(label="Model",
                               model=model,
                               series_collection=series_collection,
                               predicted_feature="y"),
    ])

    fig_name = tmp_path / "ref_vs_pred"
    results.plot_ref_vs_pred(fig_name=str(fig_name), title=False)

    output = fig_name.with_suffix(".png")
    if output.exists():
        output.unlink()


def test_prediction_results_plot_hist(tmp_path):
    series_collection = SeriesCollection([
        StateSeries([State({"x": np.array([1.0]), "y": np.array([2.0])})]),
        StateSeries([State({"x": np.array([2.0]), "y": np.array([4.0])})]),
        StateSeries([State({"x": np.array([3.0]), "y": np.array([6.0])})]),
    ])
    model = DummyStrategy()
    results = PredictionResults([
        PredictionResults.Spec(label="Model",
                               model=model,
                               series_collection=series_collection,
                               predicted_feature="y"),
    ])

    fig_name = tmp_path / "hist"
    results.plot_hist(fig_name=str(fig_name))

    output = fig_name.with_suffix(".png")
    if output.exists():
        output.unlink()


@pytest.mark.parametrize("subplots", [False, True])
@pytest.mark.parametrize("spec_count", [1, 3])
@pytest.mark.parametrize("plot_method", ["plot_ref_vs_pred", "plot_hist"])
def test_prediction_results_plot_panels(monkeypatch, tmp_path, subplots, spec_count, plot_method):
    collection = SeriesCollection([
        StateSeries([State({"x": np.array([x]), "y": np.array([2 * x])})])
        for x in [1.0, 2.0, 3.0]
    ])
    results = PredictionResults([
        PredictionResults.Spec(label=f"Model {index}", model=DummyStrategy(multiplier=2.0 + index),
                               series_collection=collection, predicted_feature="y")
        for index in range(spec_count)
    ])
    saved = []
    savefig = Figure.savefig

    def capture_figure(fig, filename, **kwargs):
        saved.append(fig)
        # Exercise real output without rendering every test image at 600 DPI.
        kwargs["dpi"] = 50
        savefig(fig, filename, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    monkeypatch.setattr(DummyStrategy, "predict", lambda *args, **kwargs: pytest.fail("Predictions recomputed"))
    fig_name = tmp_path / plot_method
    getattr(results, plot_method)(fig_name=str(fig_name), subplots=subplots)

    assert fig_name.with_suffix(".png").stat().st_size > 0
    assert len(saved) == 1
    fig = saved[0]
    assert not plt.fignum_exists(fig.number)
    assert len(fig.axes) == (spec_count if subplots else 1)
    residuals = results.reference_values - results.predicted_values
    for panel_index, ax in enumerate(fig.axes):
        indices = [panel_index] if subplots else list(range(spec_count))
        assert [text.get_text() for text in ax.get_legend().get_texts()][:len(indices)] == [
            results.labels[index] for index in indices
        ]
        if subplots:
            assert ax.get_title() == results.labels[panel_index]
        np.testing.assert_allclose(ax.get_xlim(), fig.axes[0].get_xlim())
        np.testing.assert_allclose(ax.get_ylim(), fig.axes[0].get_ylim())
        if plot_method == "plot_ref_vs_pred":
            assert len(ax.lines) == len(indices) + 5  # Points, reference, and two pairs of error bands.
            for line, index in zip(ax.lines, indices):
                np.testing.assert_allclose(line.get_xdata(), results.reference_values[:, index])
                np.testing.assert_allclose(line.get_ydata(), results.predicted_values[:, index])
            reference = ax.lines[len(indices)]
            np.testing.assert_allclose(reference.get_xdata(), reference.get_ydata())
            np.testing.assert_allclose(ax.get_xlim(), ax.get_ylim())
        else:
            assert len(ax.patches) == len(indices)
            max_diff = max(float(np.max(np.abs(residuals))), 1.0)
            edges = np.linspace(-max_diff, max_diff, 100)
            for patch, index in zip(ax.patches, indices):
                counts, _ = np.histogram(residuals[:, index], edges)
                vertices = patch.get_xy()
                np.testing.assert_allclose(vertices[1:-1:2, 0], edges[:-1])
                np.testing.assert_allclose(vertices[1:-1:2, 1], counts)


def test_prediction_results_print_metrics(tmp_path):
    series_collection = SeriesCollection([
        StateSeries([State({"x": np.array([1.0]), "y": np.array([2.0])})]),
        StateSeries([State({"x": np.array([2.0]), "y": np.array([4.0])})]),
        StateSeries([State({"x": np.array([3.0]), "y": np.array([6.0])})]),
    ])
    model = DummyStrategy()
    results = PredictionResults([
        PredictionResults.Spec(label="Model",
                               model=model,
                               series_collection=series_collection,
                               predicted_feature="y"),
    ])

    output_file = tmp_path / "metrics.txt"
    results.print_metrics(output_file=str(output_file))

    output = output_file.read_text(encoding="utf-8")
    assert "Avg" in output
    assert "Std" in output
    assert "RMS" in output
    assert "Max" in output
    assert "Model :" in output


def test_prediction_results_plot_sensitivities(tmp_path):
    series_collection = SeriesCollection([
        StateSeries([State({"x": np.array([1.0]), "y": np.array([2.0])})]),
        StateSeries([State({"x": np.array([2.0]), "y": np.array([4.0])})]),
        StateSeries([State({"x": np.array([3.0]), "y": np.array([6.0])})]),
    ])
    model = DummyStrategy()
    results = PredictionResults([
        PredictionResults.Spec(label="Model",
                               model=model,
                               series_collection=series_collection,
                               predicted_feature="y"),
    ])

    fig_prefix = tmp_path / "sens"
    results.plot_sensitivities({"x": NonPerturbator()},
                               number_of_perturbations=1,
                               fig_name_prefix=str(fig_prefix),
                               num_procs=1)

    output = tmp_path / "sens_Model.png"
    if output.exists():
        output.unlink()
