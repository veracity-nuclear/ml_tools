import random
from copy import deepcopy
from typing import Dict
import time
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress Tensorflow Warning Messages

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from sklearn.model_selection import train_test_split

from ml_tools import MinMaxNormalize, SeriesCollection, PredictionStrategy, GBMStrategy, \
                     NormalPerturbator, RelativeNormalPerturbator
from ml_tools.utils.plotting import plot_ref_vs_pred, plot_hist, plot_sensitivities, print_metrics, plot_corr_matrix, \
                                    plot_ice_pdp, plot_shap
from ml_tools.optimizer.optuna_strategy import OptunaStrategy
from ml_tools.optimizer.nn_search_space.nn_search_space import NNSearchSpace
from ml_tools.optimizer.nn_search_space.dense import Dense as DenseDim
from ml_tools.optimizer.nn_search_space.spatial_conv import SpatialConv as SpatialConvDim, SpatialMaxPool as SpatialMaxPoolDim
from ml_tools.optimizer.search_space import IntDimension, FloatDimension, CategoricalDimension, BoolDimension

from data_reader import DataReader


input_features = {'average_exposure'         : MinMaxNormalize(0., 60.),
                  'assembly_enrichment'      : MinMaxNormalize(0., 5.),
                  'measured_fixed_detector'  : MinMaxNormalize(0., 1.3),
                  'boron_concentration'      : MinMaxNormalize(0., 1500.)}


assembly_features = ["average_exposure", "assembly_enrichment"]

detector_features = ['measured_fixed_detector']

scalar_features = [feature for feature in input_features if feature not in assembly_features and
                                                            feature not in detector_features]

predicted_feature = 'cycle_exposure'


def main() -> None:
    """ Example code for doing a model optimization followed by comparison to other models
    """

    series_collection = DataReader.read_data(file_name = "sample.h5", num_procs = 20)

    models = {}
    models['GBM'] = GBMStrategy(input_features, predicted_feature)

    dense_layer = DenseDim(units      = IntDimension(16, 128),
                           activation = CategoricalDimension(["relu", "tanh"]))
    nn_space = NNSearchSpace(NNSearchSpace.Dimension(layers                = [dense_layer, dense_layer],
                                                     initial_learning_rate = FloatDimension(1e-4, 1e-1, log=True),
                                                     learning_decay_rate   = FloatDimension(1.0,   2.0          ),
                                                     epoch_limit           = IntDimension(   200, 3000, log=True),
                                                     convergence_criteria  = FloatDimension(1e-8, 1e-4, log=True),
                                                     convergence_patience  = IntDimension(    50,  200          ),
                                                     batch_size_log2       = IntDimension(     8,   11          )))
    dnn_opt = OptunaStrategy()
    models['DNN'] = dnn_opt.search(search_space      = nn_space,
                                   series_collection = random.sample(series_collection, 10000),
                                   num_trials        = 50,
                                   number_of_folds   = 5,
                                   output_file       = "dnn_optimizer.out",
                                   num_procs         = 20)

    with open("dnn.pkl", 'wb') as file:
        pickle.dump(models['DNN'], file)

    with open("dnn.pkl", 'rb') as file:
        models['DNN'] = pickle.load(file)



    conv = SpatialConvDim(input_shape = CategoricalDimension([(3, 3)]),
                          activation  = CategoricalDimension(["relu", "tanh"]),
                          filters     = IntDimension(4, 8),
                          kernel_size = CategoricalDimension([(2, 2)]),
                          strides     = CategoricalDimension([(1, 1)]),
                          padding     = BoolDimension([False]))
    pool = SpatialMaxPoolDim(input_shape = CategoricalDimension([(3, 3)]),
                             pool_size   = CategoricalDimension([(2, 2)]),
                             strides     = CategoricalDimension([(1, 1)]),
                             padding     = BoolDimension([False]))
    cnn_space = NNSearchSpace(NNSearchSpace.Dimension(layers                = [conv, pool],
                                                     initial_learning_rate = FloatDimension(1e-4, 1e-1, log=True),
                                                     learning_decay_rate   = FloatDimension( 1.0,  2.0          ),
                                                     epoch_limit           = IntDimension(   200, 3000, log=True),
                                                     convergence_criteria  = FloatDimension(1e-8, 1e-4, log=True),
                                                     convergence_patience  = IntDimension(    50,  200          ),
                                                     batch_size_log2       = IntDimension(     8,   11          )))
    cnn_opt = OptunaStrategy()
    models['CNN'] = cnn_opt.search(search_space      = cnn_space,
                                   series_collection = random.sample(series_collection, 10000),
                                   num_trials        = 50,
                                   number_of_folds   = 5,
                                   output_file       = "cnn_optimizer.out",
                                   num_procs         = 20)

    with open("cnn.pkl", 'wb') as file:
        pickle.dump(models['CNN'], file)

    with open("cnn.pkl", 'rb') as file:
        models['CNN'] = pickle.load(file)

    train_collection, valid_collection = train_test_split(series_collection, test_size=0.2)
    train(models, train_collection)

    models['GBM - GBM_Informed']               = deepcopy(models['GBM'])
    models['DNN - GBM_Informed']               = deepcopy(models['DNN'])
    models['CNN - GBM_Informed']               = deepcopy(models['CNN'])
    models['GBM - GBM_Informed'].biasing_model = models['GBM']
    models['DNN - GBM_Informed'].biasing_model = models['GBM']
    models['CNN - GBM_Informed'].biasing_model = models['GBM']

    train({name: model for name, model in models.items() if 'GBM_Informed' in name}, train_collection)

    evaluate(models, valid_collection)



def train(models: Dict[str, PredictionStrategy], train_collection: SeriesCollection) -> None:
    """ Basic function for training all models

    Parameters
    ----------
    models : Dict[str, PredictionStrategy]
        Models to be trained
    train_collection : SeriesCollection
        State series collection data to use for training
    """

    print('Training models...')
    for name, model in models.items():
        print(f'Training {name:s}...')
        start = time.time()
        if name in ['GBM', 'GBM - GBM_Informed']:
            train_data, test_data = train_test_split(train_collection, test_size=0.2)
            model.train(train_data, test_data)
        else:
            model.train(train_collection)
        print(f'  in {time.time()-start:.2f} seconds')


def evaluate(models: Dict[str, PredictionStrategy], valid_collection: SeriesCollection) -> None:
    """Basic function for plotting evaluation results

    Parameters
    ----------
    models : Dict[str, PredictionStrategy]
        Models to be evaluated
    valid_collection : SeriesCollection
        State series collection data to use for validation
    """

    print('Printing Results...')

    base_models            = {name: model for name, model in models.items() if not 'Informed' in name}
    gbm_informed_models    = {name: model for name, model in models.items() if 'GBM_Informed' in name}

    print('Plotting Correlation Matrix...')
    plot_corr_matrix(scalar_features, valid_collection, fig_name = "corr_matrix")

    print('Plotting Reference vs. Predicted Plots...')
    plot_ref_vs_pred(           base_models, valid_collection, title=False, fig_name = "ref_vs_pred_base")
    plot_ref_vs_pred(   gbm_informed_models, valid_collection, title=False, fig_name = "ref_vs_pred_gbm_informed")


    print('Plotting Histograms Plots...')
    plot_hist(           base_models, valid_collection, fig_name = "hist_base")
    plot_hist(   gbm_informed_models, valid_collection, fig_name = "hist_gbm_informed")

    print('Plotting ICE/PDP Plots...')
    plot_ice_pdp(gbm_informed_models, valid_collection, "boron_concentration", num_points=1000, fig_name_prefix = "ice_pdp_boron"    , num_procs=20)

    print('Plotting SHAP Plots...')
    feature_plots = {"scalars"         : scalar_features,
                     "avg_exp"         : ["average_exposure"],
                     "assem_enr"       : ["assembly_enrichment"],
                     "incore_det"      : ["measured_fixed_detector"]}
    plot_shap(gbm_informed_models, valid_collection, feature_plots, num_samples=1000, num_procs=20)

    print_metrics(models, valid_collection)

    perturbators = {'measured_fixed_detector' : RelativeNormalPerturbator(0.026),
                    'assembly_enrichment'     : NormalPerturbator(0.00005),
                    'average_exposure'        : RelativeNormalPerturbator(0.02),
                    'boron_concentration'     : NormalPerturbator(5.0)}

    pert_collection = random.sample(valid_collection, 100)

    print('Plotting Sensitivities Plots...')
    plot_sensitivities(models                  = base_models,
                       series_collection       = pert_collection,
                       perturbators            = perturbators,
                       number_of_perturbations = 100,
                       num_procs               = 30,
                       fig_name_prefix         = "box_plot_base")

    plot_sensitivities(models                  = gbm_informed_models,
                       series_collection       = pert_collection,
                       perturbators            = perturbators,
                       number_of_perturbations = 100,
                       num_procs               = 30,
                       fig_name_prefix         = "box_plot_informed")


if __name__ == "__main__":
    main()
