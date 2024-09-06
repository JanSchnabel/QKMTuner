import joblib
import os
import warnings
import numpy as np
import pandas as pd
from typing import Union, Optional
from functools import partial

# Optuna imports
import optuna
from optuna import Trial, Study
from optuna.trial import FrozenTrial
from optuna.samplers import BaseSampler, TPESampler
from optuna.pruners import BasePruner

# scikit-learn imports
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    PowerTransformer
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score, 
    f1_score
)

# sQUlearn imports
from squlearn.util import Executor
from squlearn.encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from squlearn.encoding_circuit import ChebyshevPQC
from squlearn.observables.observable_base import ObservableBase
from squlearn.kernel import FidelityKernel, ProjectedQuantumKernel
from squlearn.kernel.matrix.projected_quantum_kernel import OuterKernelBase
from squlearn.kernel import QSVC, QSVR, QKRR
from squlearn.kernel.optimization import KernelOptimizer, TargetAlignment
from squlearn.optimizers.optimizer_base import OptimizerBase

from .utils.scalers import BandwidthScaler, LabelMinMaxScaler, CustomLabelMinMaxScaler
from .utils.circuits import z_encoding_circuit, zz_encoding_circuit

Scaler = (
    StandardScaler |
    MinMaxScaler |
    MaxAbsScaler |
    RobustScaler |
    PowerTransformer |
    BandwidthScaler |
    LabelMinMaxScaler |
    CustomLabelMinMaxScaler |
    None
)

QuantumKernel = (
    FidelityKernel |
    ProjectedQuantumKernel
)

QuantumKernelMethod = (
    QKRR |
    QSVR |
    QSVC
)

class QKMTuner:
    """
    This program combines the respective functionalities of sQUlearm, optuna and scikit-learn
    to set up a extensive grid-search for quantum kernel methods (QKMs). As such, it supports
    fidelity quantum kernels (FQKs) and projected quantum kernels (PQKs) with QKRR, QSVR and
    QSVC approaches.

    For all sQUlearn functionalities we refer to:
     [1] The documentation: https://squlearn.github.io/index.html
     [2] The Github repository: https://github.com/sQUlearn/squlearn
     [3] The paper: https://arxiv.org/abs/2311.08990 

    Args:
        xtrain (np.ndarray): Training features of the respective dataset
        xtest (np.ndarray): Test features of the respective dataset
        ytrain (np.ndarray): Training labels/targets of the respective dataset
        ytest (np.ndarray): Test labels/targets of the respective dataset
        scaler_method (Union[Scaler, None]): Pre-scaling method for the features. Can be either 
            one of scikit-learn's preprocssing methods 
            (cf. https://scikit-learn.org/stable/api/sklearn.preprocessing.html) or one of the 
            scaler defined in .utils.scalers
        optimize_scaler (bool): Whether to optimize the feature ranges of the chosen scaler 
            within the optuna hyperparameter optimization or not (default: False)
        label_scaler (Scaler, None): Pre-scaling method for the targets. Can be either 
            one of scikit-learn's preprocssing methods 
            (cf. https://scikit-learn.org/stable/api/sklearn.preprocessing.html) or one of the 
            scaler defined in .utils.scalers
        quantum_kernel (str): A string specifying which quantum kernel to use, can be either "FQK"
            or "PQK".
        quantum_kernel_method (str): A string specifying which QKM to use. Possible values are:
            "QKRR", "QSVR", and "QSVC"
        clf_scoring (Optional[str]): A string specifying the scoring method to be used for 
            classification tasks within the objective function leveraged by the optuna 
            hyperparmater optimiation (default: "roc_auc"). See 
            https://scikit-learn.org/stable/api/sklearn.metrics.html for details.
        reg_scoring (Optional[str]): A string specifying the scoring method to be used for 
            regression tasks within the objective function leveraged by the optuna 
            hyperparmater optimiation (default: "neg_mean_squared_error"). See 
            https://scikit-learn.org/stable/api/sklearn.metrics.html for details.
        executor (Executor): The Executor of sQUlearn, see Refs. [1-3] for details. Default:
            Executor("pennylane")
        initial_parameters (Union[np.ndarray, None]): Initial parameters for the encoding circuit 
            (default: None)
        parameter_seed(Union[int, None]): Seed for the random number generator for the parameter
            initialization, if initial_parameters is None.
    """
    def __init__(
        self,
        xtrain: np.ndarray,
        xtest: np.ndarray,
        ytrain: np.ndarray,
        ytest: np.ndarray,
        scaler_method: Union[Scaler, None] = None,
        optimize_scaler: bool = False,
        label_scaler: Union[Scaler, None] = None, # only for regression tasks
        quantum_kernel: str = "FQK",
        quantum_kernel_method: str = "QKRR",
        clf_scoring: Optional[str] = "roc_auc",
        reg_scoring: Optional[str] = "neg_mean_squared_error",
        executor: Executor = Executor("pennylane"),
        initial_parameters: Union[np.ndarray, None] = None,
        parameter_seed: Union[int, None] = 0
    ) -> None:
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest
        self._scaler_method = scaler_method
        self._optimize_scaler = optimize_scaler
        self._label_scaler = label_scaler
        # get number of features in dataset
        self.num_features = self.xtrain.shape[1]

        self._quantum_kernel = quantum_kernel
        self._quantum_kernel_method = quantum_kernel_method
        self._clf_scoring = clf_scoring
        self._reg_scoring = reg_scoring
        self._executor = executor
        self._initial_parameters = initial_parameters
        self._parameter_seed = parameter_seed

        # further class variables
        self._measurement = "XYZ"
        self._outer_kernel = "gaussian"
        self._best_model = False
        self.optimal_parameters = None
        self.optimal_encoding_circuit = None
        self.qkernel_opt = None
        self.best_trial = None
        self.kernel_opt_result = None

        # set seed used for pseudorandom number generation in KFold
        self._seed = 0
        # random number generator used in QSVC
        self.rng = np.random.default_rng(self._seed)

        # raise UserWarning w.r.t different Optuna versions
        warnings.warn(
            """Optuna does not support saving/reloading across different Optuna versions with pickle.
            Thus, make sure to pip install the same versions or consider using RDBs to 
            save/reload a study accross different Optuna versions, cf.
            https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html#rdb"""
        )
    
    def save_to_dict(self, data_saving_dict=None, *args, **kwargs):
        """Helper function to save results to dict"""
        if data_saving_dict is None:
            data_saving_dict = {}
        for arg in args:
            data_saving_dict[arg] = arg
        data_saving_dict.update(kwargs)
        return data_saving_dict
    
    def reduce_function_name(self, partial_func: partial):
        function_name = partial_func.func.__name__
        gate = partial_func.keywords.get('gate', '')
        reduced_name = f"{function_name}_{gate}"
        return reduced_name
    
    def evaluate_best_model(
        self,
        encoding_circuits: list[EncodingCircuitBase],
        measurement: Union[str, ObservableBase, list] = "XYZ",
        outer_kernel: Union[str, OuterKernelBase] = "gaussian",
        num_qubits_max: int = 10,
        num_layers_max: int = 10,
        optuna_sampler: Optional[Union[BaseSampler, None]] = None,
        optuna_pruner: Optional[Union[BasePruner, None]] = None,
        n_trials: int = 100,
        n_jobs: int = 1,
        outdir: Union[str, None] = None,
        file_identifier: Union[str, None] = None #e.g., dataset_id
    ):  
        """
        Add doc.
        """
        self._measurement = measurement
        self._outer_kernel = outer_kernel
        self._best_model = True

        save_results = []
        encoding_circuits_strs = []
        for enc_circ in encoding_circuits:
            if type(enc_circ) == partial:
                #enc_circ_str = str(enc_circ).split(".")[-1].replace("'>", "").replace(")", "")
                enc_circ_str = self.reduce_function_name(enc_circ)
            elif enc_circ == z_encoding_circuit:
                enc_circ_str = str(enc_circ).split()[1]
            elif enc_circ == zz_encoding_circuit:
                enc_circ_str = str(enc_circ).split()[1]
            else:
                enc_circ_str = str(enc_circ).split(".")[-1].split(" ")[0].replace("'>","")
            # append enc_circ_str to list to reuse them later as indices in pd.DataFrame
            encoding_circuits_strs.append(enc_circ_str)
            # create study
            study_name = f"optuna_study_evaluate_best_model_{self._quantum_kernel_method}_{self._quantum_kernel}_{enc_circ_str}_{file_identifier}"
            optuna_study = optuna.create_study(study_name=study_name, sampler=optuna_sampler, pruner=optuna_pruner, direction='maximize')
            optuna_study.optimize(
                partial(self.objective, encoding_circuit=enc_circ, num_qubits=num_qubits_max,
                        num_layers=num_layers_max, best_model=self._best_model),
                n_trials=n_trials, n_jobs=n_jobs
            )
            # Save study to pickle => can be reused in KTA optimization
            path_cache_dir = str(outdir) + "cache_optuna_studies_evaluate_best_model/"
            if not os.path.exists(path_cache_dir):
                os.makedirs(path_cache_dir)
            pickle_filename = os.path.join(os.path.dirname(path_cache_dir),study_name + '.pkl')
            joblib.dump(optuna_study, pickle_filename)
            # reobtain circuit since initial encoding circuit has been optimized by optuna
            opt_circ = self.get_optimal_encoding_circuit(optuna_study, enc_circ)
            # save optimal circuit to pickle => can be reused in KTA optimization
            circ_filename = "optimal_circuit_from_" + study_name + ".pkl"
            opt_circ_file = os.path.join(os.path.dirname(path_cache_dir), circ_filename)
            joblib.dump(opt_circ, opt_circ_file)
            
            # get model prediction
            ytrain_pred, ytest_pred, best_model_ktrain, best_model_ktesttrain = self.get_model_prediction(optuna_study, opt_circ)

            # reassign class variables to best parameters found during optuna optimization
            self.optimal_encoding_circuit = opt_circ
            self.best_trial = optuna_study.best_trial
            # encoding_circuit=enc_circ_str,
            results_dict = self.save_to_dict(
                best_params=optuna_study.best_params,
                best_trial=optuna_study.best_trial,
                best_obj_val=optuna_study.best_value,
                best_feature_range = self.instantiate_model(self.best_trial, opt_circ)["scaler"].feature_range,
                ktrain=best_model_ktrain,
                ktesttrain=best_model_ktesttrain,
                ypred_train=ytrain_pred,
                ypred_test=ytest_pred
            )
            # compute metrics
            if self._label_scaler:
                ytrain = self._label_scaler.fit_transform(self.ytrain.reshape(-1,1))
                ytest = self._label_scaler.transform(self.ytest.reshape(-1,1))
            else:
                ytrain = self.ytrain
                ytest = self.ytest
            if self._quantum_kernel_method == "QSVC":
                results_dict = self.save_to_dict(
                    results_dict,
                    acc_score_train=accuracy_score(ytrain, ytrain_pred),
                    roc_auc_score_train=roc_auc_score(ytrain, ytrain_pred),
                    f1_score_train=f1_score(ytrain, ytrain_pred),
                    acc_score_test=accuracy_score(ytest, ytest_pred),
                    roc_auc_score_test=roc_auc_score(ytest, ytest_pred),
                    f1_score_test=f1_score(ytest, ytest_pred)
                )
            else:
                results_dict = self.save_to_dict(
                    results_dict,
                    mse_train=mean_squared_error(ytrain, ytrain_pred),
                    rmse_train=root_mean_squared_error(ytrain, ytrain_pred),
                    mae_train=mean_absolute_error(ytrain, ytrain_pred),
                    r2_train=r2_score(ytrain, ytrain_pred),
                    mse_test=mean_squared_error(ytest, ytest_pred),
                    rmse_test=root_mean_squared_error(ytest, ytest_pred),
                    mae_test=mean_absolute_error(ytest, ytest_pred),
                    r2_test=r2_score(ytest, ytest_pred),
                )
            save_results.append(results_dict)

            path_outdir = str(outdir) + 'results_evaluate_best_model/'
            if not os.path.exists(path_outdir):
                os.makedirs(path_outdir)
            outfile = os.path.join(os.path.dirname(path_outdir), f"results_evaluate_best_model_study_{file_identifier}_{self._quantum_kernel_method}_{self._quantum_kernel}_{enc_circ_str}.pkl")
            df = pd.DataFrame([results_dict], index=[enc_circ_str])
            df.to_pickle(outfile)
        outfile = os.path.join(os.path.dirname(path_outdir), f"results_evaluate_best_model_study_{file_identifier}_{self._quantum_kernel_method}_{self._quantum_kernel}_summary.pkl")
        df = pd.DataFrame(save_results, index=encoding_circuits_strs)
        df.to_pickle(outfile)
    
    def evaluate_grid(
        self,
        encoding_circuits: list[EncodingCircuitBase],
        measurement: Union[str, ObservableBase, list] = "XYZ",
        outer_kernel: Union[str, OuterKernelBase] = "gaussian",
        qubits_list: list[int] = [1,2,3],
        layers_list: list[int] = [1,2,3],
        optuna_sampler: Optional[Union[BaseSampler, None]] = None,
        optuna_pruner: Optional[Union[BasePruner, None]] = None,
        n_trials: int = 10,
        n_jobs: int = 1,
        outdir: Union[str, None] = None,
        file_identifier: Union[str, None] = None #e.g., dataset_id
    ):
        """
        Doc.
        
        
        Note that self.optimal_encoding_circuit and self.best_trial, which are subsequently used in 
        run_quantum_kernel_optimization() and evaluate_model_from_optimized_qkernel, correspond to 
        the best parameters found during optuna optimization for last encoding circuit in list 
        encoding_circuits.
        """
        self._measurement = measurement
        self._outer_kernel = outer_kernel
        self._best_model = False

        save_results = []
        save_kernels = []
        encoding_circuits_str = []
        # create cache dir to save optuna studies
        path_cache_dir = str(outdir) + "cache_optuna_studies_evaluate_grid/"
        if not os.path.exists(path_cache_dir):
            os.makedirs(path_cache_dir)
        for enc_circ in encoding_circuits:
            if type(enc_circ) == partial:
                #enc_circ_str = str(enc_circ).split(".")[-1].replace("'>", "").replace(")", "")
                enc_circ_str = self.reduce_function_name(enc_circ)
            elif enc_circ == z_encoding_circuit:
                enc_circ_str = str(enc_circ).split()[1]
            elif enc_circ == zz_encoding_circuit:
                enc_circ_str = str(enc_circ).split()[1]
            else:
                enc_circ_str = str(enc_circ).split(".")[-1].split(" ")[0].replace("'>","")
            encoding_circuits_str.append(enc_circ_str)
            # initialize matrices to save optuna study results and model performance scores
            mat_best_params = np.empty((len(qubits_list), len(layers_list)), dtype=dict)
            mat_best_trial = np.empty((len(qubits_list), len(layers_list)), dtype=dict)
            mat_best_obj_val = np.empty((len(qubits_list), len(layers_list)), dtype=float)
            mat_feature_range = np.empty((len(qubits_list), len(layers_list)), dtype=tuple)
            mat_r2 = np.empty((len(qubits_list), len(layers_list)))
            mat_mse = np.empty((len(qubits_list), len(layers_list)))
            mat_rmse = np.empty((len(qubits_list), len(layers_list)))
            mat_mae = np.empty((len(qubits_list), len(layers_list)))
            mat_r2_train = np.empty((len(qubits_list), len(layers_list)))
            mat_mse_train = np.empty((len(qubits_list), len(layers_list)))
            mat_rmse_train = np.empty((len(qubits_list), len(layers_list)))
            mat_mae_train = np.empty((len(qubits_list), len(layers_list)))
            mat_acc_score = np.empty((len(qubits_list), len(layers_list)))
            mat_roc_auc_score = np.empty((len(qubits_list), len(layers_list)))
            mat_f1_score = np.empty((len(qubits_list), len(layers_list)))
            mat_acc_score_train = np.empty((len(qubits_list), len(layers_list)))
            mat_roc_auc_score_train = np.empty((len(qubits_list), len(layers_list)))
            mat_f1_score_train = np.empty((len(qubits_list), len(layers_list)))
            for i, num_qubits in enumerate(qubits_list):
                for j, num_layers in enumerate(layers_list):
                    
                    study_name = f"optuna_study_evaluate_grid_{self._quantum_kernel_method}_{self._quantum_kernel}_{enc_circ_str}_num_qubits{num_qubits}_num_layers{num_layers}_{file_identifier}"
                    optuna_study = optuna.create_study(study_name=study_name, sampler=optuna_sampler, pruner=optuna_pruner, direction='maximize')
                    optuna_study.optimize(
                        partial(self.objective, encoding_circuit=enc_circ, num_qubits=num_qubits,
                                num_layers=num_layers, best_model=self._best_model),
                        n_trials=n_trials, n_jobs=n_jobs
                    )
                    # Save each study to pickle
                    pickle_filename = os.path.join(os.path.dirname(path_cache_dir), study_name + ".pkl")
                    joblib.dump(optuna_study, pickle_filename)
                    # set circuit to the current num_qubits and num_layers
                    circuit = self.instantiate_encoding_circuit(optuna_study.best_trial, enc_circ, num_qubits, num_layers, self._best_model)
                    ytrain_pred, ytest_pred, best_model_ktrain, best_model_ktesttrain = self.get_model_prediction(optuna_study, circuit)
                    
                    # save kernels
                    kernels_dict = self.save_to_dict(
                        encoding_circuit=enc_circ_str,
                        num_qubits=num_qubits,
                        num_layers=num_layers,
                        ktrain=best_model_ktrain,
                        ktesttrain=best_model_ktesttrain
                    )
                    save_kernels.append(kernels_dict)
                    # write some optuna StudySummary to matrix
                    mat_best_params[i,j] = optuna_study.best_params
                    mat_best_trial[i,j] = optuna_study.best_trial
                    mat_best_obj_val[i,j] = optuna_study.best_value
                    mat_feature_range[i,j] = self.instantiate_model(optuna_study.best_trial, circuit)["scaler"].feature_range
                    # compute metrics
                    if self._label_scaler:
                        ytrain = self._label_scaler.fit_transform(self.ytrain.reshape(-1,1))
                        ytest = self._label_scaler.transform(self.ytest.reshape(-1,1))
                    else:
                        ytrain = self.ytrain
                        ytest = self.ytest
                    if self._quantum_kernel_method == "QSVC":
                        mat_acc_score_train[i,j] = accuracy_score(ytrain, ytrain_pred)
                        mat_roc_auc_score_train[i,j] = roc_auc_score(ytrain, ytrain_pred)
                        mat_f1_score_train[i,j] = f1_score(ytrain, ytrain_pred)
                        mat_acc_score[i,j] = accuracy_score(ytest, ytest_pred)
                        mat_roc_auc_score[i,j] = roc_auc_score(ytest, ytest_pred)
                        mat_f1_score[i,j] = f1_score(ytest, ytest_pred)
                    else:
                        mat_r2_train[i,j] = r2_score(ytrain, ytrain_pred)
                        mat_mse_train[i,j] = mean_squared_error(ytrain, ytrain_pred)
                        mat_rmse_train[i,j] = root_mean_squared_error(ytrain, ytrain_pred)
                        mat_mae_train[i,j] = mean_absolute_error(ytrain, ytrain_pred)
                        mat_r2[i,j] = r2_score(ytest, ytest_pred)
                        mat_mse[i,j] = mean_squared_error(ytest, ytest_pred)
                        mat_rmse[i,j] = root_mean_squared_error(ytest, ytest_pred)
                        mat_mae[i,j] = mean_absolute_error(ytest, ytest_pred)
            
            #save results to dict
            results_dict = self.save_to_dict(
                best_param_mat = mat_best_params,
                best_trial_mat = mat_best_trial,
                best_objective_value_mat = mat_best_obj_val,
                feature_range_mat = mat_feature_range
            )
            if self._quantum_kernel_method == "QSVC":
                results_dict = self.save_to_dict(
                    results_dict,
                    acc_score_train_mat = mat_acc_score_train,
                    roc_auc_score_train_mat = mat_roc_auc_score_train,
                    f1_score_train_mat = mat_f1_score_train,
                    acc_score_test_mat = mat_acc_score,
                    roc_auc_score_test_mat = mat_roc_auc_score,
                    f1_score_test_mat = mat_f1_score
                )
                # get best model for enc_circ
                idx_opt = self._get_pos_of_max_matrix_element(mat_roc_auc_score)
            else:
                results_dict = self.save_to_dict(
                    results_dict,
                    mse_train_mat = mat_mse_train,
                    rmse_train_mat = mat_rmse_train,
                    mae_train_mat = mat_mae_train,
                    r2_train_mat = mat_r2_train,
                    mse_test_mat = mat_mse,
                    rmse_test_mat = mat_rmse,
                    mae_test_mat = mat_mae,
                    r2_test_mat = mat_r2
                )
                # get best model for enc_circ
                idx_opt = self._get_pos_of_max_matrix_element(mat_r2)
            save_results.append(results_dict)

            # define optimal num_qubits and num_layers for optimal encoding circuit
            num_qubits_opt = qubits_list[idx_opt[0]]
            num_layers_opt = layers_list[idx_opt[1]]
            self.best_trial = mat_best_trial[idx_opt]
            self.optimal_encoding_circuit = self.instantiate_encoding_circuit(self.best_trial, enc_circ, num_qubits_opt, num_layers_opt, self._best_model)
            # save optimal circuit and optimal trial to file
            path_opt_grid = str(outdir) + "cache_optuna_studies_evaluate_grid/optimal/"
            if not os.path.exists(path_opt_grid):
                os.makedirs(path_opt_grid)
            file_optimal_encoding_circuit = os.path.join(os.path.dirname(path_opt_grid), f"optimal_{enc_circ_str}_from_evaluate_grid_{file_identifier}_{self._quantum_kernel}_{self._quantum_kernel_method}.pkl")
            file_best_trial = os.path.join(os.path.dirname(path_opt_grid), f"best_trial_{enc_circ_str}_from_evaluate_grid_{file_identifier}_{self._quantum_kernel}_{self._quantum_kernel_method}.pkl")
            joblib.dump(self.optimal_encoding_circuit, file_optimal_encoding_circuit)
            joblib.dump(self.best_trial, file_best_trial)
        
            # Save all results, i.e. save_results and save_kernels to *.pkl-files
            path_results = str(outdir) + "results_evaluate_grid/"
            if not os.path.exists(path_results):
                os.makedirs(path_results)
            outfile_params = os.path.join(os.path.dirname(path_results), f"results_evaluate_grid_study_{file_identifier}_{self._quantum_kernel_method}_{self._quantum_kernel}_{enc_circ_str}.pkl")
            outfile_kernel = os.path.join(os.path.dirname(path_results), f"kernels_evaluate_grid_study_{file_identifier}_{self._quantum_kernel_method}_{self._quantum_kernel}_{enc_circ_str}.pkl")
            df1 = pd.DataFrame([results_dict], index=[enc_circ_str])
            df1.to_pickle(outfile_params)
            df2 = pd.DataFrame(save_kernels)
            df2.to_pickle(outfile_kernel)
        # if run ends successfully, create summary of everything and store to .pkl-file
        outfile_params = os.path.join(os.path.dirname(path_results), f"results_evaluate_grid_study_{file_identifier}_{self._quantum_kernel_method}_{self._quantum_kernel}_summary.pkl")
        outfile_kernel = os.path.join(os.path.dirname(path_results), f"kernels_evaluate_grid_study_{file_identifier}_{self._quantum_kernel_method}_{self._quantum_kernel}_summary.pkl")
        df1 = pd.DataFrame(save_results, index=encoding_circuits_str)
        df1.to_pickle(outfile_params)
        df2 = pd.DataFrame(save_kernels)
        df2.to_pickle(outfile_kernel)
    
    def _get_pos_of_max_matrix_element(self, matrix: np.ndarray) -> tuple:
        """Helper function to find largest element in matrix"""
        idx_pos = np.unravel_index(np.argmax(matrix), matrix.shape)
        return idx_pos

    def run_quantum_kernel_optimization(
        self,
        optimizer: OptimizerBase,
        best_model: Union[FrozenTrial, str, None] = None,
        encoding_circuit: Union[EncodingCircuitBase, str, None] = None # this has to be of the form, e.g., YZ_CX_EncodingCircuit(num_qubits=4, num_features=1, num_layers=2)
    ):
        """
        Note that
            best_model = None
            encoding_circuit = None
        only works if evalate_best_model() or evaluate_grid() have been executed 
        with one encoding circuit.
        """
        if not best_model:
            best_trial = self.best_trial
        elif isinstance(best_model, str):
            study = joblib.load(best_model)
            best_trial = study.best_trial
        elif isinstance(best_model, FrozenTrial):
            best_trial = best_model
        else:
            raise ValueError(
                "best_model must be either a *.pkl-file containing the Study object or a FrozenTrial representing study.best_trial,"
                "or evaluate_best_model()/evaluate_grid() for one encoding circuit has to be called."
            )
        # assign best_trial to class variable
        self.best_trial = best_trial
        # get model corresponding to best_trial
        if not encoding_circuit:
            opt_model = self.instantiate_model(self.best_trial, self.optimal_encoding_circuit)
            qkernel = self.instantiate_quantum_kernel(self.best_trial, self.optimal_encoding_circuit)
        elif isinstance(encoding_circuit, str):
            opt_circ = joblib.load(encoding_circuit)
            self.optimal_encoding_circuit = opt_circ
            opt_model = self.instantiate_model(self.best_trial, opt_circ)
            qkernel = self.instantiate_quantum_kernel(self.best_trial, opt_circ)
        elif isinstance(encoding_circuit, EncodingCircuitBase):
            opt_model = self.instantiate_model(self.best_trial, encoding_circuit)
            qkernel = self.instantiate_quantum_kernel(self.best_trial, encoding_circuit)
            self.optimal_encoding_circuit = encoding_circuit
        else:
            raise ValueError(
                "encoding_circuit must be either a *.pkl-file, an EncodingCircuitBase object"
                "or evaluate_best_model()/evaluate_grid() has to be called beforehand"
            )
        scaler = opt_model["scaler"]
        # rescale data with scaler from model
        if scaler:
            x_train = scaler.fit_transform(self.xtrain)
        else:
            x_train = self.xtrain
        if self._label_scaler:
            y_train = self._label_scaler.fit_transform(self.ytrain.reshape(-1,1))
        else:
            y_train = self.ytrain
        # set up kernel optimization
        kernel_loss = TargetAlignment(quantum_kernel=qkernel)
        kernel_optimizer = KernelOptimizer(loss=kernel_loss, optimizer=optimizer)
        # run the optimization
        kernel_opt_result = kernel_optimizer.run_optimization(X=x_train, y=y_train)
        self.kernel_opt_result = kernel_opt_result
        # get the best parameters from the optimization
        self.optimal_parameters = kernel_opt_result.x
        print(self.optimal_parameters)
        self.qkernel_opt = qkernel.assign_parameters(self.optimal_parameters)
        return self

    def evaluate_model_from_optimized_qkernel(
        self,
        optuna_sampler: Optional[Union[BaseSampler, None]] = None,
        optuna_pruner: Optional[Union[BasePruner, None]] = None,
        n_trials: int = 100,
        n_jobs: int = 1,
        outdir: Union[str, None] = None,
        file_identifier: Union[str, None] = None #e.g., dataset_id
    ):
        self._initial_parameters = self.optimal_parameters
        self._optimize_scaler = False
        
        save_results = []
        # get optimal model corresponding to optimized parameters
        opt_model = self.instantiate_model_for_prediction(self.best_trial, self.optimal_encoding_circuit)
        #print(opt_model["estimator"].get_params("quantum_kernel")["quantum_kernel"].parameters)
        # evaluate performance of only optimized qkernel
        opt_model.fit(self.xtrain, self.ytrain)
        ytest_pred_opt_only = opt_model.predict(self.xtest)
        scaler = opt_model["scaler"]
        if scaler:
            x_train = scaler.fit_transform(self.xtrain)
            x_test = scaler.transform(self.xtest)
        else:
            x_train = self.xtrain
            x_test = self.xtest
        # compute kernels corresponding to opt_model
        if self._label_scaler:
            ktrain_opt_only = opt_model["estimator"].regressor_.get_params("quantum_kernel")["quantum_kernel"].evaluate(x_train)
            ktesttrain_opt_only = opt_model["estimator"].regressor_.get_params("quantum_kernel")["quantum_kernel"].evaluate(x_test, x_train)
        else:
            ktrain_opt_only = opt_model["estimator"].get_params("quantum_kernel")["quantum_kernel"].evaluate(x_train)
            ktesttrain_opt_only = opt_model["estimator"].get_params("quantum_kernel")["quantum_kernel"].evaluate(x_test, x_train)
        # apply optuna hyperparameter optimization on optimized qkernel
        optuna_study = optuna.create_study(sampler=optuna_sampler, pruner=optuna_pruner, direction='maximize')
        optuna_study.optimize(
            partial(self.objective_optimized_kernel, encoding_circuit=self.optimal_encoding_circuit),
            n_trials=n_trials, n_jobs=n_jobs
        )
        ytrain_pred, ytest_pred, ktrain, ktesttrain = self.get_model_prediction(optuna_study, self.optimal_encoding_circuit)
        # save everything to dict
        results_dict = self.save_to_dict(
            optimal_parameters = self.optimal_parameters,
            ktrain_opt_only = ktrain_opt_only,
            ktesttrain_opt_only = ktesttrain_opt_only,
            best_optuna_params_new = optuna_study.best_params,
            best_feature_range = scaler.feature_range,
            ktrain_opt_and_optuna = ktrain,
            ktesttrain_opt_and_optuna = ktesttrain
        )
        # compute metrics
        if self._label_scaler:
            ytrain = self._label_scaler.fit_transform(self.ytrain.reshape(-1,1))
            ytest = self._label_scaler.transform(self.ytest.reshape(-1,1))
        else:
            ytrain = self.ytrain
            ytest = self.ytest
        if self._quantum_kernel_method == "QSVC":
            results_dict = self.save_to_dict(
                results_dict,
                acc_score_test_opt_only = accuracy_score(ytest, ytest_pred_opt_only),
                roc_auc_score_test_opt_only = roc_auc_score(ytest, ytest_pred_opt_only),
                f1_score_test_mat_opt_only = f1_score(ytest, ytest_pred_opt_only),
                acc_score_test_opt_and_optuna = accuracy_score(ytest, ytest_pred),
                roc_auc_score_opt_and_optun = roc_auc_score(ytest, ytest_pred),
                f1_score_opt_and_optuna = f1_score(ytest, ytest_pred)
            )
        else:
            results_dict = self.save_to_dict(
                results_dict,
                mse_test_opt_only = mean_squared_error(ytest,ytest_pred_opt_only),
                rmse_test_opt_only = root_mean_squared_error(ytest,ytest_pred_opt_only),
                mae_test_opt_only = mean_absolute_error(ytest,ytest_pred_opt_only),
                r2_test_opt_only = r2_score(ytest,ytest_pred_opt_only),
                mse_test_opt_and_optuna = mean_squared_error(ytest, ytest_pred),
                rmse_test_opt_and_optuna = root_mean_squared_error(ytest, ytest_pred),
                mae_test_opt_and_optuna = mean_absolute_error(ytest, ytest_pred),
                r2_test_opt_and_optuna = r2_score(ytest, ytest_pred)
            )
        save_results.append(results_dict)
        # Save all results
        path_results = str(outdir) + "results_evaluate_model_from_optimized_qkernel/"
        if not os.path.exists(path_results):
            os.makedirs(path_results)
        outfile = os.path.join(os.path.dirname(path_results), f"results_evaluate_model_from_optimized_qkernel_{self._quantum_kernel_method}_{self._quantum_kernel}_{file_identifier}.pkl")
        #df = pd.DataFrame.from_records(save_results)
        df = pd.DataFrame(save_results)
        return df.to_pickle(outfile)
        #return df.to_csv(outfile)
    
    def objective(
        self,
        trial: Trial,
        encoding_circuit: EncodingCircuitBase,
        num_qubits: int,
        num_layers: int,
        best_model: bool
    ) -> float:
        circuit = self.instantiate_encoding_circuit(trial, encoding_circuit, num_qubits, num_layers, best_model)
        model = self.instantiate_model(trial, circuit)

        # # Instantiate KFold object to allow to reproducibly separate between
        # train and test splits for cross-validation
        if self._quantum_kernel_method == 'QSVC':
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self._seed)
            scores = cross_val_score(model, self.xtrain, self.ytrain, scoring=self._clf_scoring, cv=skf, error_score='raise')
        else:
            kf = KFold(n_splits=5, shuffle=True, random_state=self._seed)
            scores = cross_val_score(model, self.xtrain, self.ytrain, scoring=self._reg_scoring, cv=kf, error_score='raise')
        
        # Mean values tend to be strongly influenced by outliers, so if a single
        # split has a very good performance, but the others don't, then the score
        # will be artificially high. Similarly, the mean is also influenced by
        # very small values, while the median won't. Such, the minimum between
        # the mean and median will prevent scores form becoming too optimistic, 
        # which may negatively influence surrogate-model optimization (e.g. Bayesian)
        # methods. This also helps prevent overfitting towards an "easy" fold.
        # From: https://medium.com/@walter_sperat/using-optuna-with-sklearn-the-right-way-part-1-6b4ad0ab2451
        return np.min([np.mean(scores), np.median(scores)], axis=0)
        #return self._chop_float(np.min([np.mean(scores), np.median(scores)], axis=0))
    
    def objective_optimized_kernel(
        self,
        trial: Trial,
        encoding_circuit: EncodingCircuitBase
    ) -> float:
        model = self.instantiate_model(trial, encoding_circuit)
        
        # Instantiate KFold object to allow to reproducibly separate between
        # train and test splits for cross-validation
        if self._quantum_kernel_method == 'QSVC':
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self._seed)
            scores = cross_val_score(model, self.xtrain, self.ytrain, scoring=self._clf_scoring, cv=skf, error_score='raise')
        else:
            kf = KFold(n_splits=5, shuffle=True, random_state=self._seed)
            scores = cross_val_score(model, self.xtrain, self.ytrain, scoring=self._reg_scoring, cv=kf, error_score='raise')
        return np.min([np.mean(scores), np.median(scores)], axis=0)

    def instantiate_model(
        self,
        trial: Trial,
        encoding_circuit: EncodingCircuitBase
        ) -> Pipeline:
        if self._label_scaler:
            from sklearn.compose import TransformedTargetRegressor
            if isinstance(encoding_circuit, ChebyshevPQC):
                model = Pipeline([
                    ('scaler', self.instantiate_scaler(trial, chebyshev=True)),
                    ('estimator', TransformedTargetRegressor(self.instantiate_quantum_kernel_method(trial, encoding_circuit), transformer=self._label_scaler, check_inverse=False))
                ])
            else:
                model = Pipeline([
                    ('scaler', self.instantiate_scaler(trial)),
                    ('estimator', TransformedTargetRegressor(self.instantiate_quantum_kernel_method(trial, encoding_circuit), transformer=self._label_scaler, check_inverse=False))
                ])
        else:
            if isinstance(encoding_circuit, ChebyshevPQC):
                model = Pipeline([
                    ('scaler', self.instantiate_scaler(trial, chebyshev=True)),
                    ('estimator', self.instantiate_quantum_kernel_method(trial, encoding_circuit))
                ])
            else:
                model = Pipeline([
                    ('scaler', self.instantiate_scaler(trial)),
                    ('estimator', self.instantiate_quantum_kernel_method(trial, encoding_circuit))
                ])
        return model
    
    def instantiate_model_for_prediction(
        self,
        trial: Trial,
        encoding_circuit: EncodingCircuitBase
        ) -> Pipeline:
        if self._label_scaler:
            from sklearn.compose import TransformedTargetRegressor
            if isinstance(encoding_circuit, ChebyshevPQC):
                model = Pipeline([
                    ('scaler', self.instantiate_scaler(trial, chebyshev=True)),
                    ('estimator', TransformedTargetRegressor(self.instantiate_quantum_kernel_method(trial, encoding_circuit), transformer=CustomLabelMinMaxScaler(), check_inverse=False))
                ])
            else:
                model = Pipeline([
                    ('scaler', self.instantiate_scaler(trial)),
                    ('estimator', TransformedTargetRegressor(self.instantiate_quantum_kernel_method(trial, encoding_circuit), transformer=CustomLabelMinMaxScaler(), check_inverse=False))
                ])
        else:
            if isinstance(encoding_circuit, ChebyshevPQC):
                model = Pipeline([
                    ('scaler', self.instantiate_scaler(trial, chebyshev=True)),
                    ('estimator', self.instantiate_quantum_kernel_method(trial, encoding_circuit))
                ])
            else:
                model = Pipeline([
                    ('scaler', self.instantiate_scaler(trial)),
                    ('estimator', self.instantiate_quantum_kernel_method(trial, encoding_circuit))
                ])
        return model

    # only for evaluate_best_model()
    def get_optimal_encoding_circuit(self, study: Study, encoding_circuit: EncodingCircuitBase) -> EncodingCircuitBase:
        try:
            study.best_params["num_qubits"]
            num_qubits_opt = study.best_params["num_qubits"]
            num_layers_opt = study.best_params["num_layers"]
        except:
            num_qubits_opt = self.num_features
            num_layers_opt = study.best_params["num_layers"]
        circuit_opt = encoding_circuit(num_qubits_opt, self.num_features, num_layers_opt)
        return circuit_opt
    
    def get_model_prediction(
        self,
        optuna_result: Union[Study, FrozenTrial],
        encoding_circuit: EncodingCircuitBase
    ):
        if isinstance(optuna_result, Study):
            best_trial = optuna_result.best_trial
        elif isinstance(optuna_result, FrozenTrial):
            best_trial = optuna_result
        else:
            raise ValueError("optuna_result has to be of type Study or FrozenTrial.")
        opt_model = self.instantiate_model_for_prediction(best_trial, encoding_circuit)
        # fit optimal model
        opt_model.fit(self.xtrain, self.ytrain)
        # make predictions with optimal model
        ytrain_pred = opt_model.predict(self.xtrain)
        ytest_pred = opt_model.predict(self.xtest)
        # get optimal scaler from pipeline for qkernel.evaluate()
        scaler = opt_model["scaler"]
        if scaler:
            x_train = scaler.fit_transform(self.xtrain)
            x_test = scaler.transform(self.xtest)
        else:
            x_train = self.xtrain
            x_test = self.xtest
        # replace this by instantiate_quantum_kernel()
        if self._label_scaler:
            best_model_ktrain = opt_model["estimator"].regressor_.get_params("quantum_kernel")["quantum_kernel"].evaluate(x_train)
            best_model_ktesttrain = opt_model["estimator"].regressor_.get_params("quantum_kernel")["quantum_kernel"].evaluate(x_test, x_train)
        else:
            best_model_ktrain = opt_model["estimator"].get_params("quantum_kernel")["quantum_kernel"].evaluate(x_train)
            best_model_ktesttrain = opt_model["estimator"].get_params("quantum_kernel")["quantum_kernel"].evaluate(x_test, x_train)
        
        return ytrain_pred, ytest_pred, best_model_ktrain, best_model_ktesttrain

    def instantiate_encoding_circuit(
        self,
        trial: Trial,
        encoding_circuit: EncodingCircuitBase,
        num_qubits_max: int,
        num_layers_max: int,
        best_model: bool
    ):
        # if num_qubits_max % self.num_features != 0:
        #     raise ValueError(f"num_qubits_max have to be a multiple of num_features = {self.num_features}!")
        
        num_qubits_min = self.num_features
        if best_model:
            if num_qubits_min == num_qubits_max:
                num_layers = trial.suggest_int('num_layers', 1, num_layers_max, log=True)
                num_qubits = num_qubits_max
            else:
                num_qubits = trial.suggest_int('num_qubits', num_qubits_min, num_qubits_max, step=self.num_features)
                num_layers = trial.suggest_int('num_layers', 1, num_layers_max, log=True)
            circuit = encoding_circuit(num_qubits, self.num_features, num_layers)
        else:
            circuit = encoding_circuit(num_qubits_max, self.num_features, num_layers_max)
        return circuit

    def instantiate_quantum_kernel_method(
        self,
        trial: Trial,
        encoding_circuit: EncodingCircuitBase
    ) -> QuantumKernelMethod:
        
        if self._quantum_kernel_method == 'QKRR':
            qk_estimator = self.instantiate_qkrr(trial, encoding_circuit)
        elif self._quantum_kernel_method == 'QSVR':
            qk_estimator = self.instantiate_qsvr(trial, encoding_circuit)
        elif self._quantum_kernel_method == 'QSVC':
            qk_estimator = self.instantiate_qsvc(trial, encoding_circuit)
        return qk_estimator
    
    def instantiate_quantum_kernel(
        self,
        trial: Trial,
        encoding_circuit: EncodingCircuitBase,
    ) -> QuantumKernel:
        
        if self._quantum_kernel == "FQK":
            qkernel = self.instantiate_fqk(encoding_circuit)
        elif self._quantum_kernel == "PQK":
            qkernel = self.instantiate_pqk(trial, encoding_circuit)
        
        return qkernel

    def instantiate_max_abs_scaler(self):
        _scaler_method = MaxAbsScaler()
        return _scaler_method

    def instantiate_min_max_scaler(self, trial: Trial, chebyshev: bool = False):
        if self._optimize_scaler:
            if chebyshev:
                min_range = trial.suggest_float('min_range', -1.0, 0.0)
                max_range = trial.suggest_float('max_range', 0.0, 1.0)
                feature_range = (min_range, max_range)
                params = {'clip': True}
            else:
                min_range = trial.suggest_float('min_range', -1.* np.pi / 2., 0.0)
                max_range = trial.suggest_float('max_range', 0.0, np.pi / 2.)
                feature_range = (min_range, max_range)
                params  = {'clip': False}
            _scaler_method = MinMaxScaler(feature_range=feature_range, **params)
        else:
            if chebyshev:
            #    _scaler_method = MinMaxScaler(feature_range=(-1,1), clip=True)
                _scaler_method = self._scaler_method
            else:
                _scaler_method = self._scaler_method
        return _scaler_method
        
    def instantiate_standard_scaler(self, trial: Trial):
        if self._optimize_scaler:
            params = {
                'with_mean': trial.suggest_categorical(
                    'with_mean', [True, False]
                ),
                'with_std': trial.suggest_categorical(
                    'with_std', [True, False]
                )
            }
            _scaler_method = StandardScaler(**params)
        return _scaler_method

    def instantiate_robust_scaler(self, trial: Trial):
        if self._optimize_scaler:
            params = {
                'with_centering': trial.suggest_categorical(
                    'with_centering', [True, False]
                ),
                'with_scaling': trial.suggest_categorical(
                    'with_scaling', [True, False]
                ),
                'unit_variance': trial.suggest_categorical(
                    'unit_variance', [True, False]
                )
            }
            _scaler_method = RobustScaler(**params, quantile_range=(25.0, 75.0))
        return _scaler_method
    
    def instantiate_power_transformer(self, trial: Trial):
        if self._optimize_scaler:
            params = {
                'method': trial.suggest_categorical(
                    'method', ['yeo-johnson', 'box-cox']
                ),
                'standardize': trial.suggest_categorical(
                    'standardize', [True, False]
                )
            }
            _scaler_method = PowerTransformer(**params)
        return _scaler_method
    
    def instantiate_bandwidth_scaler(self, trial: Trial):
        if self._optimize_scaler:
            params = {
                'bandwidth': trial.suggest_float(
                    'bandwidth', 0.1, 1.0, log=True
                )
            }
            _scaler_method = BandwidthScaler(**params)
        return _scaler_method
        
    def instantiate_scaler(self, trial: Trial, chebyshev: bool=False) -> Scaler:
        if not self._scaler_method:
            scaler = None
        elif isinstance(self._scaler_method, StandardScaler):
            scaler = self.instantiate_standard_scaler(trial)
        elif isinstance(self._scaler_method, MinMaxScaler):
            scaler = self.instantiate_min_max_scaler(trial, chebyshev)
        elif isinstance(self._scaler_method, MaxAbsScaler):
            scaler = self.instantiate_max_abs_scaler()
        elif isinstance(self._scaler_method, RobustScaler):
            scaler = self.instantiate_robust_scaler(trial)
        elif isinstance(self._scaler_method, PowerTransformer):
            scaler = self.instantiate_power_transformer(trial)
        elif isinstance(self._scaler_method, BandwidthScaler):
            scaler = self.instantiate_bandwidth_scaler(trial)
        return scaler

    def instantiate_fqk(self, encoding_circuit: EncodingCircuitBase):
        fqk = FidelityKernel(
            encoding_circuit=encoding_circuit,
            executor=self._executor,
            initial_parameters=self._initial_parameters,
            parameter_seed=self._parameter_seed
        )
        return fqk
    
    def instantiate_pqk_gaussian(self, trial: Trial, encoding_circuit: EncodingCircuitBase) -> ProjectedQuantumKernel:
        self._outer_kernel = "gaussian"
        gamma = trial.suggest_float('gamma', 1.e-3, 1.e3, log=True)
        pqk = self._get_pqk(encoding_circuit, gamma=gamma)
        return pqk

    def instantiate_pqk_matern(self, trial: Trial, encoding_circuit: EncodingCircuitBase, nu: float=1.5) -> ProjectedQuantumKernel:
        self._outer_kernel = "matern"
        if nu not in [0.5, 1.5, 2.5]:
            raise ValueError("Computational cost for nu not in [0.5, 1.5, 2.5] approx. 10 times higher")
        length_scale = trial.suggest_float('length_scale', 1.e-5, 1.e5, log=True)
        pqk = self._get_pqk(encoding_circuit, length_scale=length_scale, nu=nu)
        return pqk

    def instantiate_pqk_exp_sine_squared(self, trial: Trial, encoding_circuit: EncodingCircuitBase) -> ProjectedQuantumKernel:
        self._outer_kernel = "expsinesquared"
        length_scale = trial.suggest_float('length_scale', 1.e-3, 1.e3, log=True)
        periodicity = trial.suggest_float('periodicity', 1.e-3, 1.e3, log=True)
        pqk = self._get_pqk(encoding_circuit, length_scale=length_scale, periodicity=periodicity)
        return pqk

    def instantiate_pqk_rationale_quadratic(self, trial: Trial, encoding_circuit: EncodingCircuitBase) -> ProjectedQuantumKernel:
        self._outer_kernel = "rationalquadratic"
        alpha = trial.suggest_float('alpha', 1.e-5, 1.e5, log=True)
        length_scale = trial.suggest_float('length_scale', 1.e-5, 1.e5, log=True)
        pqk = self._get_pqk(encoding_circuit, length_scale=length_scale, alpha=alpha)
        return pqk

    def instantiate_pqk_dot_product(self, trial: Trial, encoding_circuit: EncodingCircuitBase) -> ProjectedQuantumKernel:
        self._outer_kernel = "dotproduct"
        sigma_0 = trial.suggest_float('sigma_0', 1.e-3, 1.e3, log=True)
        pqk = self._get_pqk(encoding_circuit,  sigma_0=sigma_0)
        return pqk

    def instantiate_pqk_pairwise_kernel(self, trial: Trial, encoding_circuit: EncodingCircuitBase) -> ProjectedQuantumKernel:
        self._outer_kernel = "pairwisekernel"
        gamma = trial.suggest_float('gamma', 1.e-3, 1.e3, log=True)
        metric = trial.suggest_categorical(
            'metric', ['linear', 'additive_chi2', 'chi2', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine']
        )
        pqk = self._get_pqk(encoding_circuit, gamma=gamma, metric=metric)
        return pqk

    def instantiate_pqk(self, trial: Trial, encoding_circuit: EncodingCircuitBase) -> ProjectedQuantumKernel:
        if self._outer_kernel == "gaussian":
            pqk = self.instantiate_pqk_gaussian(trial, encoding_circuit)
        elif self._outer_kernel == "matern":
            pqk = self.instantiate_pqk_matern(trial, encoding_circuit)
        elif self._outer_kernel == "expsinesquared":
            pqk = self.instantiate_pqk_exp_sine_squared(trial, encoding_circuit)
        elif self._outer_kernel == "rationalquadratic":
            pqk = self.instantiate_pqk_rationale_quadratic(trial, encoding_circuit)
        elif self._outer_kernel == "dotproduct":
            pqk = self.instantiate_pqk_dot_product(trial, encoding_circuit)
        elif self._outer_kernel == "pairwisekernel":
            pqk = self.instantiate_pqk_pairwise_kernel(trial, encoding_circuit)
        return pqk

    def _get_pqk(self, encoding_circuit: EncodingCircuitBase, **kwargs):
        kernel = ProjectedQuantumKernel(
            encoding_circuit=encoding_circuit,
            executor=self._executor,
            measurement=self._measurement,
            outer_kernel=self._outer_kernel,
            initial_parameters=self._initial_parameters,
            parameter_seed=self._parameter_seed,
            **kwargs
        )
        return kernel

    def instantiate_qkrr(
        self,
        trial: Trial,
        encoding_circuit: EncodingCircuitBase
    ) -> QKRR:
     
        alpha = trial.suggest_float('alpha', 1.e-10, 1.e2, log=True)
        qkernel = self.instantiate_quantum_kernel(trial, encoding_circuit)
        qkrr_inst = QKRR(
            quantum_kernel=qkernel,
            alpha=alpha
        )

        return qkrr_inst

    def instantiate_qsvr(
        self,
        trial: Trial,
        encoding_circuit: EncodingCircuitBase
    ) -> QSVR:
        C = trial.suggest_float('C', 1.e-2, 1.e7, log=True)
        epsilon = trial.suggest_float('epsilon', 1.e-9, 1.0, log=True)
        qkernel = self.instantiate_quantum_kernel(trial, encoding_circuit)
        qsvr_inst = QSVR(
            quantum_kernel=qkernel,
            C=C,
            epsilon=epsilon
        )
        return qsvr_inst

    def instantiate_qsvc(
        self,
        trial: Trial,
        encoding_circuit: EncodingCircuitBase,
    ) -> QSVC:
        C = trial.suggest_float('C', 1.e-2, 1.e7, log=True)
        qkernel = self.instantiate_quantum_kernel(trial, encoding_circuit)
        qsvc_inst = QSVC(
            quantum_kernel=qkernel,
            C=C,
            probability=True,
            random_state=self.rng.integers(100000)
        )
        return qsvc_inst
    
    def _chop_float(self, number: float) -> float:
        formatted_number = "{:.5g}".format(number)
        return float(formatted_number)
