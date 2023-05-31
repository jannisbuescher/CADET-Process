import torch

from CADETProcess.nn.training import FCNN

import numpy as np
import os
from CADETProcess.optimization import OptimizerBase


class NNInterface(OptimizerBase):
    """Wrapper around nn."""

    def run(self, optimization_problem, c_experiment):
        """Solve optimization problem using functional nn implementation.

        Parameters
        ----------
        optimization_problem : OptimizationProblem
            DESCRIPTION.

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

        model = FCNN(901,2)

        state_dict = torch.load('./CADETProcess/nn/model_weights/model100.sdict')
        model.load_state_dict(state_dict)

        x = torch.tensor(c_experiment, dtype=torch.float32)

        best_parameters = model(x)

        self.results.exit_flag = 0
        self.results.exit_message = "We are great!"

        x = best_parameters.detach().numpy()
        f = optimization_problem.evaluate_objectives(x)
        #g = optimization_problem.evaluate_nonlinear_constraints(x)
        #cv = optimization_problem.evaluate_nonlinear_constraints_violation(x)

        self.run_post_evaluation_processing(x, f, None, None, 1)

        return self.results
