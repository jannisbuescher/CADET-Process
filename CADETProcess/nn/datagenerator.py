import random
import numpy as np

from CADETProcess.CADETProcessError import CADETProcessError
from CADETProcess.optimization.optimizationProblem import OptimizationProblem

from CADETProcess.dataStructure.nested_dict import get_nested_value

from copy import deepcopy

from addict import Dict

import matplotlib.pyplot as plt

from CADETProcess.processModel import Process
from CADETProcess.processModel import FlowSheet
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import Inlet, Outlet, LumpedRateModelWithPores

rng = np.random.default_rng(1)

class DataGenerator:
    """
    Generates practice data from given optimization problem
    """

    def __init__(self, optimization_problem, simulator, process, seed=None):
        # Check OptimizationProblem
        if not isinstance(optimization_problem, OptimizationProblem):
            raise TypeError('Expected OptimizationProblem')
        self.op = optimization_problem
        self.sim = simulator
        self.process = deepcopy(process)  # deepcopy

    def generate(self, nb_examples):
        vars = self.op.independent_variables
        data = np.ndarray((nb_examples, len(vars) + 901))

        for i in range(nb_examples):
            #param = DataGenerator._sample(high, low)  # illegal
            # param = self.op.create_initial_values(
            #     1, method='random', seed=rng.integers(1,1000000)
            # )[0]
            param = DataGenerator._sample(len(vars))


            # for j, var in enumerate(vars):
            #     path = var.parameter_path
            #     name = var.name
            #     value = param[j]
            process = DataGenerator._get_process(self.op.untransform(np.array(param).reshape(1,-1))[0])
            
            #self.process.parameters.update(param_update) # nested_dict / addict / parameter path
            simulation_results = self.sim.simulate(process)
            sim_data = deepcopy(simulation_results.solution.outlet.outlet.solution)

            data[i, 0:len(vars)] = param
            data[i, len(vars):] = deepcopy(sim_data.reshape((-1,)))

        return data
    
    @staticmethod
    def _sample(n):
        return rng.uniform(low=0, high=1.0, size=n)
    
    @staticmethod
    def _get_process(param):

        bed_porosity = param[0]
        axial_dispersion = param[1]

        component_system = ComponentSystem(['Non-penetrating Tracer'])

        feed = Inlet(component_system, name='feed')
        feed.c = [0.0005]

        eluent = Inlet(component_system, name='eluent')
        eluent.c = [0]

        column = LumpedRateModelWithPores(component_system, name='column')

        # parameters, not all of which are to be optimized:
        column.length = 0.1
        column.diameter = 0.0077
        column.particle_radius = 34e-6

        column.axial_dispersion = axial_dispersion
        column.bed_porosity = bed_porosity
        column.particle_porosity = 0.8
        column.film_diffusion = [0]

        outlet = Outlet(component_system, name='outlet')

        flow_sheet = FlowSheet(component_system)

        flow_sheet.add_unit(feed)
        flow_sheet.add_unit(eluent)
        flow_sheet.add_unit(column)
        flow_sheet.add_unit(outlet)

        flow_sheet.add_connection(feed, column)
        flow_sheet.add_connection(eluent, column)
        flow_sheet.add_connection(column, outlet)

        Q_ml_min = 0.5  # ml/min
        Q_m3_s = Q_ml_min/(60*1e6)
        V_tracer = 50e-9  # m3

        process = Process(flow_sheet, 'Tracer')
        process.cycle_time = 15*60

        process.add_event(
            'feed_on',
            'flow_sheet.feed.flow_rate',
            Q_m3_s, 0
        )
        process.add_event(
            'feed_off',
            'flow_sheet.feed.flow_rate',
            0,
            V_tracer/Q_m3_s
        )

        process.add_event(
            'feed_water_on',
            'flow_sheet.eluent.flow_rate',
            Q_m3_s,
            V_tracer/Q_m3_s
        )

        process.add_event(
            'eluent_off',
            'flow_sheet.eluent.flow_rate',
            0,
            process.cycle_time
        )

        return process