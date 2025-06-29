import random
from copy import deepcopy
from typing import List, Dict
import time
import h5py as h5
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress Tensorflow Warning Messages

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from sklearn.model_selection import train_test_split

from ml_tools import MinMaxNormalize, NoProcessing, StateSeries, PredictionStrategy, GBMStrategy, NNStrategy, \
                     NormalPerturbator, RelativeNormalPerturbator
from ml_tools.utils.plotting import plot_ref_vs_pred, plot_hist, plot_sensitivities, print_metrics, plot_corr_matrix, \
                                    plot_ice_pdp, plot_shap
from ml_tools.model.nn_strategy import Dense

from data_reader import DataReader
from optimizer import Optimizer
from dnn_optimizer import DNNOptunaOptimizer
from cnn_optimizer import CNNOptunaOptimizer


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

    state_series = DataReader.read_data(file_name = "sample.h5", num_procs = 20)

    models = {}
    models['GBM'] = GBMStrategy(input_features, predicted_feature)

    search_space = DNNOptunaOptimizer.Dimensions(initial_learning_rate = (  1e-6,   1e-1),
                                                 learning_decay_rate   = (   0.0,    1.0),
                                                 batch_size_log2       = (     8,     11),
                                                 num_dens_layers       = (     1,      5),
                                                 dens_layers           = [Optimizer.Dimensions.Layer(neurons = (5, 500), activation = ['relu', 'tanh'], dropout = (0.0, 1.0)),
                                                                          Optimizer.Dimensions.Layer(neurons = (5, 500), activation = ['relu', 'tanh'], dropout = (0.0, 1.0)),
                                                                          Optimizer.Dimensions.Layer(neurons = (5, 500), activation = ['relu', 'tanh'], dropout = (0.0, 1.0)),
                                                                          Optimizer.Dimensions.Layer(neurons = (5, 500), activation = ['relu', 'tanh'], dropout = (0.0, 1.0)),
                                                                          Optimizer.Dimensions.Layer(neurons = (5, 500), activation = ['relu', 'tanh'], dropout = (0.0, 1.0))])

    optimizer = DNNOptunaOptimizer(dimensions           = search_space,
                                   input_features       = input_features,
                                   predicted_feature    = predicted_feature,
                                   state_series         = random.sample(state_series, 10000),
                                   num_procs            = 20,
                                   test_fraction        = 0.2,
                                   number_of_folds      = 5,
                                   epoch_limit          = 3000,
                                   convergence_criteria = 1E-14,
                                   convergence_patience = 200,
                                   biasing_model        = None)

    models['DNN'] = optimizer.optimize(num_trials  = 50,
                                       output_file = "dnn_optimizer.out")

    with open("dnn.pkl", 'wb') as file:
        pickle.dump(models['DNN'], file)

    with open("dnn.pkl", 'rb') as file:
        models['DNN'] = pickle.load(file)



    search_space = CNNOptunaOptimizer.Dimensions(initial_learning_rate     = (  1e-6,   1e-1),
                                                 learning_decay_rate       = (   0.0,    1.0),
                                                 batch_size_log2           = (     8,     11),
                                                 assembly_CNN_stencil_size = (     1,      3),
                                                 assembly_CNN_filters_log2 = (     0,      2),
                                                 assembly_CNN_activation   = ['relu', 'tanh'],
                                                 detector_CNN_stencil_size = (     1,      7),
                                                 detector_CNN_filters_log2 = (     0,      2),
                                                 detector_CNN_activation   = ['relu', 'tanh'],
                                                 CNN_dropout               = (   0.0,    1.0),
                                                 num_dens_layers           = (     1,      5),
                                                 dens_layers               = [Optimizer.Dimensions.Layer(neurons = (5, 500), activation = ['relu', 'tanh'], dropout = (0.0, 1.0)),
                                                                              Optimizer.Dimensions.Layer(neurons = (5, 500), activation = ['relu', 'tanh'], dropout = (0.0, 1.0)),
                                                                              Optimizer.Dimensions.Layer(neurons = (5, 500), activation = ['relu', 'tanh'], dropout = (0.0, 1.0)),
                                                                              Optimizer.Dimensions.Layer(neurons = (5, 500), activation = ['relu', 'tanh'], dropout = (0.0, 1.0)),
                                                                              Optimizer.Dimensions.Layer(neurons = (5, 500), activation = ['relu', 'tanh'], dropout = (0.0, 1.0))])

    optimizer = CNNOptunaOptimizer(dimensions           = search_space,
                                   input_features       = input_features,
                                   predicted_feature    = predicted_feature,
                                   state_series         = random.sample(state_series, 10000),
                                   num_procs            = 20,
                                   test_fraction        = 0.2,
                                   number_of_folds      = 5,
                                   epoch_limit          = 3000,
                                   convergence_criteria = 1E-14,
                                   convergence_patience = 200,
                                   biasing_model        = None)

    models['CNN'] = optimizer.optimize(num_trials  = 50,
                                       output_file = "cnn_optimizer.out")

    with open("cnn.pkl", 'wb') as file:
        pickle.dump(models['CNN'], file)

    with open("cnn.pkl", 'rb') as file:
        models['CNN'] = pickle.load(file)

    train_series, valid_series = train_test_split(state_series, test_size=0.2)
    train(models, train_series)

    models['GBM - GBM_Informed']               = deepcopy(models['GBM'])
    models['DNN - GBM_Informed']               = deepcopy(models['DNN'])
    models['CNN - GBM_Informed']               = deepcopy(models['CNN'])
    models['GBM - GBM_Informed'].biasing_model = models['GBM']
    models['DNN - GBM_Informed'].biasing_model = models['GBM']
    models['CNN - GBM_Informed'].biasing_model = models['GBM']

    train({name: model for name, model in models.items() if 'GBM_Informed' in name}, train_series)

    evaluate(models, valid_series)



def train(models: Dict[str, PredictionStrategy], train_series: List[StateSeries]) -> None:
    """ Basic function for training all models

    Parameters
    ----------
    models : Dict[str, PredictionStrategy]
        Models to be trained
    train_series : List[StateSeries]
        State series data to use for training
    """

    print('Training models...')
    for name, model in models.items():
        print(f'Training {name:s}...')
        start = time.time()
        if name in ['GBM', 'GBM - GBM_Informed']:
            train_data, test_data = train_test_split(train_series, test_size=0.2)
            model.train(train_data, test_data)
        else:
            model.train(train_series)
        print(f'  in {time.time()-start:.2f} seconds')


def evaluate(models: Dict[str, PredictionStrategy], valid_series: List[StateSeries]) -> None:
    """Basic function for plotting evaluation results

    Parameters
    ----------
    models : Dict[str, PredictionStrategy]
        Models to be evaluated
    valid_series : List[StateSeries]
        State series data to use for validation
    """

    print('Printing Results...')

    base_models            = {name: model for name, model in models.items() if not 'Informed' in name}
    gbm_informed_models    = {name: model for name, model in models.items() if 'GBM_Informed' in name}

    print('Plotting Correlation Matrix...')
    plot_corr_matrix(scalar_features, valid_series, fig_name = "corr_matrix")

    print('Plotting Reference vs. Predicted Plots...')
    plot_ref_vs_pred(           base_models, valid_series, title=False, fig_name = "ref_vs_pred_base")
    plot_ref_vs_pred(   gbm_informed_models, valid_series, title=False, fig_name = "ref_vs_pred_gbm_informed")


    print('Plotting Histograms Plots...')
    plot_hist(           base_models, valid_series, fig_name = "hist_base")
    plot_hist(   gbm_informed_models, valid_series, fig_name = "hist_gbm_informed")

    print('Plotting ICE/PDP Plots...')
    plot_ice_pdp(gbm_informed_models, valid_series, "boron_concentration", num_points=1000, fig_name_prefix = "ice_pdp_boron"    , num_procs=20)

    print('Plotting SHAP Plots...')
    feature_plots = {"scalars"         : scalar_features,
                     "avg_exp"         : ["average_exposure"],
                     "assem_enr"       : ["assembly_enrichment"],
                     "incore_det"      : ["measured_fixed_detector"]}
    plot_shap(gbm_informed_models, valid_series, feature_plots, num_samples=1000, num_procs=20)

    print_metrics(models, valid_series)

    perturbators = {'measured_fixed_detector' : RelativeNormalPerturbator(0.026),
                    'assembly_enrichment'     : NormalPerturbator(0.00005),
                    'average_exposure'        : RelativeNormalPerturbator(0.02),
                    'boron_concentration'     : NormalPerturbator(5.0)}

    pert_states = random.sample(valid_series, 100)

    print('Plotting Sensitivities Plots...')
    plot_sensitivities(models                  = base_models,
                       state_series            = pert_states,
                       perturbators            = perturbators,
                       number_of_perturbations = 100,
                       num_procs               = 30,
                       fig_name_prefix         = "box_plot_base")

    plot_sensitivities(models                  = informed_models,
                       state_series            = pert_states,
                       perturbators            = perturbators,
                       number_of_perturbations = 100,
                       num_procs               = 30,
                       fig_name_prefix         = "box_plot_informed")


if __name__ == "__main__":
    main()
