import time
import warnings

import numpy as np

from CADETProcess.dataStructure import UnsignedInteger, UnsignedFloat
from CADETProcess.optimization import OptimizerBase

from ax.service.managed_loop import optimize


class AxInterface(OptimizerBase):
    """Wrapper around ax."""

    supports_multi_objective = True
    supports_linear_constraints = True
    supports_linear_equality_constraints = True
    supports_nonlinear_constraints = True

    seed = UnsignedInteger(default=12345)
    pop_size = UnsignedInteger()
    xtol = UnsignedFloat(default=1e-8)
    cvtol = UnsignedFloat(default=1e-6)
    cv_tol = cvtol
    ftol = UnsignedFloat(default=0.0025)
    n_max_gen = UnsignedInteger()
    n_max_evals = UnsignedInteger(default=100000)
    _specific_options = [
        'seed', 'pop_size', 'xtol', 'cvtol', 'ftol', 'n_max_gen', 'n_max_evals',
    ]

    def run(self, optimization_problem, x0=None):
        """Solve optimization problem using functional ax implementation.

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

        # self.run_post_evaluation_processing([[.0,.1],[.5,.6]], [optimization_problem.evaluate_objectives([.0,.1], untransform=True),
        #                                                         optimization_problem.evaluate_objectives([.5,.6], untransform=True)]
        #                                     , None, None, 0)

        def eval_fun(parameterization):
            x = np.array([parameterization.get(f"x{i+1}") for i in range(2)])
            obj = optimization_problem.evaluate_objectives_population(x, untransform=True)
            return {"objective": (obj, 0.0)}

        best_parameters, values, experiment, model = optimize(
            parameters=[
                {
                    "name": "x1",
                    "type": "range",
                    "bounds": [0.0, 1.0],
                    "value_type": "float",  # Optional, defaults to inference from type of "bounds".
                    "log_scale": False,  # Optional, defaults to False.
                },
                {
                    "name": "x2",
                    "type": "range",
                    "bounds": [0.0, 1.0],
                }
            ],
            experiment_name="test",
            objective_name="objective",
            evaluation_function=eval_fun,
            minimize=True,  # Optional, defaults to False.
            total_trials=30, # Optional.
            arms_per_trial=2,
        )

        self.results.exit_flag = 1
        self.results.exit_message = "We are great!"

        x = list(best_parameters.values())
        f = optimization_problem.evaluate_objectives(list(best_parameters.values()), untransform=True)
        #g = optimization_problem.evaluate_nonlinear_constraints(x)
        #cv = optimization_problem.evaluate_nonlinear_constraints_violation(x)

        self.run_post_evaluation_processing(x, f, None, None, 1)

        return self.results


class LoopWrapper(AxInterface):

    def __str__(self):
        return 'LoopWrapper'
