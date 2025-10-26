from __future__ import annotations
from ml_tools.optimizer.search_space import (
    SearchSpace,
    StructDimension,
    IntDimension,
    FloatDimension,
    CategoricalDimension,
)


class GBMSearchSpace(SearchSpace):
    """Gradient Boosting Machine (LightGBM) hyperparameter search space.

    Parameters
    ----------
    dimensions : GBMSearchSpace.Dimension
        The root hyperparameter search space to explore.

    Notes
    -----
    - For parameter semantics and valid ranges, see
      ``ml_tools.model.gbm_strategy.GBMStrategy`` (docstrings reference
      the corresponding LightGBM parameters and recommended constraints).
    """

    class Dimension(StructDimension):
        """GBM hyperparameter dimensions.

        Parameters
        ----------
        boosting_type : CategoricalDimension, optional
            Boosting method; default ['gbdt'].
        objective : CategoricalDimension, optional
            Loss function; default ['regression'].
        metric : CategoricalDimension, optional
            Evaluation metric; default ['rmse'].
        num_leaves : IntDimension, optional
            Maximum number of leaves in one tree; default IntDimension(64, 64).
        learning_rate : FloatDimension, optional
            Learning/shrinkage rate; default FloatDimension(0.07, 0.07).
        n_estimators : IntDimension, optional
            Number of boosting iterations; default IntDimension(1000, 1000).
        max_depth : IntDimension, optional
            Max depth limit for tree model; default IntDimension(4, 4).
        min_child_samples : IntDimension, optional
            Minimum data in one leaf to split; default IntDimension(20, 20).
        subsample : FloatDimension, optional
            Bagging fraction; default FloatDimension(0.8, 0.8).
        colsample_bytree : FloatDimension, optional
            Feature fraction for each tree; default FloatDimension(0.8, 0.8).
        reg_alpha : FloatDimension, optional
            L1 regularization; default FloatDimension(0.0, 0.0).
        reg_lambda : FloatDimension, optional
            L2 regularization; default FloatDimension(0.0, 0.0).
        verbose : IntDimension, optional
            LightGBM verbosity; default IntDimension(-1, -1).
        num_boost_round : IntDimension, optional
            Number of boosting iterations; default IntDimension(20, 20).
        stopping_rounds : IntDimension, optional
            Early-stopping rounds; default IntDimension(5, 5).

        Attributes
        ----------
        boosting_type : CategoricalDimension
            Boosting method.
        objective : CategoricalDimension
            Loss function.
        metric : CategoricalDimension
            Evaluation metric.
        num_leaves : IntDimension
            Maximum number of leaves in one tree.
        learning_rate : FloatDimension
            Learning/shrinkage rate.
        n_estimators : IntDimension
            Number of boosting iterations.
        max_depth : IntDimension
            Max depth limit for tree model.
        min_child_samples : IntDimension
            Minimum data in one leaf to split.
        subsample : FloatDimension
            Bagging fraction.
        colsample_bytree : FloatDimension
            Feature fraction for each tree.
        reg_alpha : FloatDimension
            L1 regularization.
        reg_lambda : FloatDimension
            L2 regularization.
        verbose : IntDimension
            LightGBM verbosity.
        num_boost_round : IntDimension
            Number of boosting iterations.
        stopping_rounds : IntDimension
            Early-stopping rounds.
        """

        @property
        def boosting_type(self) -> CategoricalDimension:
            return self.fields["boosting_type"]

        @boosting_type.setter
        def boosting_type(self, value: CategoricalDimension) -> None:
            self.fields["boosting_type"] = value

        @property
        def objective(self) -> CategoricalDimension:
            return self.fields["objective"]

        @objective.setter
        def objective(self, value: CategoricalDimension) -> None:
            self.fields["objective"] = value

        @property
        def metric(self) -> CategoricalDimension:
            return self.fields["metric"]

        @metric.setter
        def metric(self, value: CategoricalDimension) -> None:
            self.fields["metric"] = value

        @property
        def num_leaves(self) -> IntDimension:
            return self.fields["num_leaves"]

        @num_leaves.setter
        def num_leaves(self, value: IntDimension) -> None:
            self.fields["num_leaves"] = value

        @property
        def learning_rate(self) -> FloatDimension:
            return self.fields["learning_rate"]

        @learning_rate.setter
        def learning_rate(self, value: FloatDimension) -> None:
            self.fields["learning_rate"] = value

        @property
        def n_estimators(self) -> IntDimension:
            return self.fields["n_estimators"]

        @n_estimators.setter
        def n_estimators(self, value: IntDimension) -> None:
            self.fields["n_estimators"] = value

        @property
        def max_depth(self) -> IntDimension:
            return self.fields["max_depth"]

        @max_depth.setter
        def max_depth(self, value: IntDimension) -> None:
            self.fields["max_depth"] = value

        @property
        def min_child_samples(self) -> IntDimension:
            return self.fields["min_child_samples"]

        @min_child_samples.setter
        def min_child_samples(self, value: IntDimension) -> None:
            self.fields["min_child_samples"] = value

        @property
        def subsample(self) -> FloatDimension:
            return self.fields["subsample"]

        @subsample.setter
        def subsample(self, value: FloatDimension) -> None:
            self.fields["subsample"] = value

        @property
        def colsample_bytree(self) -> FloatDimension:
            return self.fields["colsample_bytree"]

        @colsample_bytree.setter
        def colsample_bytree(self, value: FloatDimension) -> None:
            self.fields["colsample_bytree"] = value

        @property
        def reg_alpha(self) -> FloatDimension:
            return self.fields["reg_alpha"]

        @reg_alpha.setter
        def reg_alpha(self, value: FloatDimension) -> None:
            self.fields["reg_alpha"] = value

        @property
        def reg_lambda(self) -> FloatDimension:
            return self.fields["reg_lambda"]

        @reg_lambda.setter
        def reg_lambda(self, value: FloatDimension) -> None:
            self.fields["reg_lambda"] = value

        @property
        def verbose(self) -> IntDimension:
            return self.fields["verbose"]

        @verbose.setter
        def verbose(self, value: IntDimension) -> None:
            self.fields["verbose"] = value

        @property
        def num_boost_round(self) -> IntDimension:
            return self.fields["num_boost_round"]

        @num_boost_round.setter
        def num_boost_round(self, value: IntDimension) -> None:
            self.fields["num_boost_round"] = value

        @property
        def stopping_rounds(self) -> IntDimension:
            return self.fields["stopping_rounds"]

        @stopping_rounds.setter
        def stopping_rounds(self, value: IntDimension) -> None:
            self.fields["stopping_rounds"] = value

        def __init__(self,
                     boosting_type:     CategoricalDimension = CategoricalDimension(['gbdt']),
                     objective:         CategoricalDimension = CategoricalDimension(['regression']),
                     metric:            CategoricalDimension = CategoricalDimension(['rmse']),
                     num_leaves:        IntDimension         = IntDimension(64, 64),
                     learning_rate:     FloatDimension       = FloatDimension(0.07, 0.07),
                     n_estimators:      IntDimension         = IntDimension(1000, 1000),
                     max_depth:         IntDimension         = IntDimension(4, 4),
                     min_child_samples: IntDimension         = IntDimension(20, 20),
                     subsample:         FloatDimension       = FloatDimension(0.8, 0.8),
                     colsample_bytree:  FloatDimension       = FloatDimension(0.8, 0.8),
                     reg_alpha:         FloatDimension       = FloatDimension(0.0, 0.0),
                     reg_lambda:        FloatDimension       = FloatDimension(0.0, 0.0),
                     verbose:           IntDimension         = IntDimension(-1, -1),
                     num_boost_round:   IntDimension         = IntDimension(20, 20),
                     stopping_rounds:   IntDimension         = IntDimension(5, 5)) -> None:

            self.fields = {}
            self.boosting_type     = boosting_type
            self.objective         = objective
            self.metric            = metric
            self.num_leaves        = num_leaves
            self.learning_rate     = learning_rate
            self.n_estimators      = n_estimators
            self.max_depth         = max_depth
            self.min_child_samples = min_child_samples
            self.subsample         = subsample
            self.colsample_bytree  = colsample_bytree
            self.reg_alpha         = reg_alpha
            self.reg_lambda        = reg_lambda
            self.verbose           = verbose
            self.num_boost_round   = num_boost_round
            self.stopping_rounds   = stopping_rounds

            super().__init__(self.fields)

    def __init__(self, dimensions: StructDimension) -> None:
        assert isinstance(dimensions, GBMSearchSpace.Dimension), (
            f"dimensions must be a GBMSearchSpace.Dimension, got {type(dimensions)}"
        )
        super().__init__(prediction_strategy_type="GBMStrategy", dimensions=dimensions)
