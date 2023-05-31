import torch

from CADETProcess.nn.training import FCNN, train_fcnn

import numpy as np
import os
from CADETProcess.optimization import OptimizerBase


class NNInterface(OptimizerBase):
    """Wrapper around nn."""

    def run(self, optimization_problem, c_experiment=None):
        """Solve optimization problem using functional nn implementation.

        Parameters
        ----------
        optimization_problem : OptimizationProblem
        c_experiment: np.array
            The experiment concentration if the length does not fit the length of the simulation

        Returns
        -------
        results : OptimizationResults
            Optimization results including optimization_problem and solver
            configuration.

        See Also
        --------
        evaluate_objectives
        options

        """
        if c_experiment is None:
            refIO_super = optimization_problem.objectives[0].objective.references
            refIO = list(refIO_super.values())[0]
            c_experiment = refIO.solution_original.reshape(-1) # only if size fits

        model_path = './CADETProcess/nn/model_weights/model1000'
        data_path = './CADETProcess/nn/data/data_1000.npy'

        if not os.path.isfile(model_path):
            NNInterface._train_model(901, 2, model_path, data_path)
        
        model = FCNN.load(901, 2, model_path)

        x = torch.tensor(c_experiment, dtype=torch.float32)

        best_parameters = model(x)

        self.results.exit_flag = 0
        self.results.exit_message = "We are great!"

        x = best_parameters[0].detach().numpy()
        f = optimization_problem.evaluate_objectives(x, untransform=True)
        #g = optimization_problem.evaluate_nonlinear_constraints(x)
        #cv = optimization_problem.evaluate_nonlinear_constraints_violation(x)

        self.run_post_evaluation_processing(x, f, None, None, 1)

        return self.results
    
    @staticmethod
    def _train_model(input_dim, output_dim, model_path, data_path):
        train_fcnn(input_dim, output_dim, model_path, data_path)
