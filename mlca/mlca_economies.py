"""
FILE DESCRIPTION:

This file implements the class Economies. This class is used for simulating the ICA, i.e., the MLCA mechanism.
Specifically, this class stores all the information about the economies in the iterative procedure of the ML-based preference elicitation algorithm (Algorithm 1 in Section 2).

Economies has the following functionalities:
    0.CONSTRUCTOR: __init__(self,  SATS_auction_instance, SATS_auction_instance_seed, Qinit, Qmax, Qround, scaler)
        SATS_auction_instance = auction instance created with SATS (GSVM, LSVM, MRVM)
        SATS_auction_instance_seed = seed of SATS auction instance
        Qinit = number of intial data points, i.e., bundle-value pairs (queried randomly per bidder prior to the elicitation procedure) (positive integer)
        Qmax = maximal number of possible value queries per bidder (positive integer)
        Qround = number of sampled marginals per bidder per auction round == number of value queries per bidder per auction round
        scaler = scaler is a instance from sklearn.MinMaxScaler() used for scaling the values of the bundle-value pairs
        This method creates a instance of the class Economies. Details about the attributes can be found below.
    1.METHOD: get_info(self, final_summary=False)
        This method retunrs information of the current status of the MLCA run.
    2.METHOD: get_number_of_elicited_bids(self, bidder=None)
        This method gets the number of elicited bids so far
    3.METHOD: calculate_efficiency_per_iteration(self):
        Calculates efficient allocation per acution round and corresponding efficiency
    4.METHOD: set_NN_parameters(self, parameters)
    5.METHOD: set_MIP_parameters(self, parameters)
    6.METHOD: set_initial_bids(self, initial_bids=None, fitted_scaler=None)
        initial_bids = self defined initial bids
        fitted_scaler = corresponding fitted scaler instance
        This method creates sets of initial bundle-value pairs, sets the elicited bids attribute and the scaler attribute, i.e., bids.
        If not self defnied initial_bids are given it calls the function mlca_util.initial_bids_mlca_unif (uniformly at random from the bundle space 2^m).
    4.METHOD: reset_argmax_allocations(self)
        This method cleares the argmax allocation a^(t) from the round t.
    5.METHOD: reset_current_query_profile(self)
        This method resets the current query profile S
    6.METHOD: reset_NN_models(self)
        This method resets the NN keras models, i.e., the DNNs from the previous round.
    7.METHOD: reset_economy_status(self)
        This method resets the economy staus, if an unrestricted allocation has already been calculated for this economy
    8.METHOD: solve_SATS_auction_instance(self)
        This method solves thegiven SATS instance for the true efficient allocation.
    9.METHOD: sample_marginal_economies(self, active_bidder, number_of_marginals)
        This method samples randomly marginal economies for the active_bidder
    10.METHOD: update_elicited_bids(self)
        This method adds the bids from the current query profile to the elicited bids
    11.METHOD: update_current_query_profile(self, bidder, bundle_to_add)
        This method  updates the current query profile for a bidder with a bundle_to_add
    12.METHOD: value_queries(self, bidder_id, bundles)
        This method performs a value query for a bidder with bidder_id and several bundles
    13.METHOD: check_bundle_contained(self, bundle, bidder)
        This method checks if the bundle already has been queried from bidder in the elicited bids or the current query profile.
    14.METHOD: next_queries(self, economy_key, active_bidder)
        This method performs the nextQueries algorithm for the bidder active_bidder and the economy economy_key
    15a.METHOD: estimation_step_economy(self, economy_key)
        This method performs the estimation step for a by economy_key specified economy
    15b.METHOD: estimation_step(self)
        This method performs the estimation step for all bidders at once
    16.METHOD: optimization_step(self, economy_key, bidder_specific_constraints=None)
        This method performs the optimization step for a by a economy_key specified economy.
        If bidder_specific_constraints is not None this solves the additional bidder specific MIP with constraints given by the elicited bids and the current query profile of the specific bidder.
        bidder_specific_constraints is a dict with (bidder name, np.array of specific bundles that serve as constraints)
    17.METHOD: calculate_mlca_allocation(self, economy='Main Economy')
        This method calculates the MLCA allocation given elicited bids for any economy speified.
    18.METHOD: solve_WDP(self, elicited_bids, verbose=0)
        This method solves the WDP  given elicited_bids.
    19.METHOD: calculate_efficiency_of_allocation(self, allocation, allocation_scw, verbose=0)
        This method calculates the efficiency w.r.t to the true efficient SATS allocation
    20.METHOD: calculate_vcg_payments(self, forced_recalc=False)
        This method calculates the VCG-style payments given current elicited bids.
"""

import copy
import itertools
import json
import logging
import multiprocessing
# Libs
import os
import random
import signal
import sys
import time
import uuid
from collections import OrderedDict, defaultdict
from datetime import timedelta, datetime
from functools import partial

import docplex
import numpy as np
import torch
from joblib import Parallel, delayed
from mlca.mlca_mvnn import MLCA_MVNN
from mlca.mlca_util import key_to_int, format_solution_mip_new, initial_bids_mlca_unif
from mlca.mlca_util import timediff_d_h_m_s
from mlca.mlca_wdp import MLCA_WDP
from mlca.mvnn_mip_torch_new import MVNN_MIP_TORCH_NEW
from numpyencoder import NumpyEncoder


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def globalize(func):
    def result(*args, **kwargs):
        return func(*args, **kwargs)

    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result


def parallel_apply_along_axis(func1d, axis, arr, num_cpus, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, num_cpus)]

    pool = multiprocessing.Pool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)


def unpacking_apply_along_axis(all_args):
    (func1d, axis, arr, args, kwargs) = all_args
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)


def merge_dicts(*dict_args):
    """
    Given any number of dictionaries, shallow copy and merge into a new dict,
    precedence goes to key-value pairs in latter dictionaries.
    """
    dict_args = dict_args[0]
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def eval_bidder_nn(bidder,
                   seed,
                   fitted_scaler,
                   local_scaling_factor,
                   NN_parameters,
                   elicited_bids,
                   num_cpu_per_job
                   ):
    torch.set_num_threads(num_cpu_per_job)
    bids = elicited_bids[bidder]

    start = time.time()
    # instantiate class
    mvnn = MLCA_MVNN(X_train=bids[0],
                     Y_train=bids[1],
                     scaler=fitted_scaler,
                     local_scaling_factor=local_scaling_factor
                     )

    # initialize model
    mvnn.initialize_model(model_parameters=NN_parameters[bidder])

    # fit model to data
    train_logs = mvnn.fit(epochs=NN_parameters[bidder]['epochs'],
                          batch_size=NN_parameters[bidder]['batch_size'],
                          seed=seed,
                          X_valid=None,
                          Y_valid=None,
                          bidder_id=key_to_int(bidder))
    end = time.time()
    logging.info('Time for ' + bidder + ': %s sec\n', round(end - start))
    return {bidder: [mvnn, train_logs]}


class MLCA_Economies:

    def __init__(self,
                 SATS_auction_instance,
                 SATS_auction_instance_seed,
                 Qinit, Qmax, Qround,
                 separate_economy_training,
                 new_query_option,
                 balanced_global_marginals,
                 parallelize_training,
                 scaler,
                 local_scaling_factor,
                 start_time):

        # STATIC ATTRIBUTES
        self.SATS_auction_instance = SATS_auction_instance  # auction instance from SATS: LSVM, GSVM or MRVM generated via PySats.py.
        self.SATS_auction_instance_allocation = None  # true efficient allocation of auction instance
        self.SATS_auction_instance_scw = None  # SATS_auction_instance.get_efficient_allocation()[1]  # social welfare of true efficient allocation of auction instance
        self.SATS_auction_instance_seed = SATS_auction_instance_seed  # auction instance seed from SATS
        self.bidder_ids = list(SATS_auction_instance.get_bidder_ids())  # bidder ids in this auction instance.
        self.bidder_names = list('Bidder_{}'.format(bidder_id) for bidder_id in self.bidder_ids)
        self.N = len(self.bidder_ids)  # number of bidders
        self.good_ids = set(SATS_auction_instance.get_good_ids())  # good ids in this auction instance
        self.M = len(self.good_ids)  # number of items
        self.Qinit = Qinit  # number of intial data points, i.e., bundle-value pairs (queried randomly per bidder prior to the elicitation procedure, different per bidder)
        self.Qmax = Qmax  # maximal number of possible value queries in the preference elicitation algorithm (PEA) per bidder
        if Qround >= 1 and Qround <= self.N:
            self.Qround = Qround  # maximal number of marginal economies per auction round per bidder
        else:
            raise ValueError(f'Selected Qround:{Qround} not in [1,...,{self.N}].')
        self.scaler = scaler  # scaler is a instance from sklearn.MinMaxScaler() used for scaling the values of the bundle-value pairs
        self.fitted_scaler = None  # fitted scaler to the initial bids
        self.mlca_allocation = None  # mlca allocation
        self.mlca_scw = None  # true social welfare of mlca allocation
        self.mlca_allocation_efficiency = None  # efficiency of mlca allocation
        self.MIP_parameters = None  # MIP parameters
        self.mlca_iteration = 0  # mlca iteration tracker
        self.revenue = 0  # sum of mlca payments
        self.relative_revenue = None  # relative revenue cp to SATS_auction_instance_scw
        self.number_of_optimization = {'normal': 0,
                                       'bidder_specific_MIP': 0,
                                       "bidder_specific_RS": 0,
                                       "bidder_specific_RS_REUSED": 0,
                                       "bidder_specific_DE": 0,
                                       "bidder_specific_DE_REUSED": 0,
                                       }
        self.total_time_elapsed_distr = {'MIP': None,
                                         'ML_TRAIN': None,
                                         'OTHER': None
                                         }
        self.local_scaling_factor = local_scaling_factor
        self.separate_economy_training = separate_economy_training
        self.new_query_option = new_query_option
        self.balanced_global_marginals = balanced_global_marginals
        self.parallelize_training = parallelize_training

        self.start_time = start_time
        self.end_time = None

        subsets = list(map(list, itertools.combinations(self.bidder_ids,
                                                        self.N - 1)))  # orderedDict containing all economies and the corresponding bidder ids of the bidders which are active in these economies.
        subsets.sort(reverse=True)  # sort s.t. Marginal Economy (-0)-> Marginal Economy (-1) ->...
        self.economies = OrderedDict(list(('Marginal Economy -({})'.format(i), econ) for econ, i in zip(subsets, [
            [x for x in self.bidder_ids if x not in subset][0] for subset in subsets])))
        self.economies['Main Economy'] = self.bidder_ids
        self.economies_names = OrderedDict(list((key, ['Bidder_{}'.format(s) for s in value]) for key, value in
                                                self.economies.items()))  # orderedDict containing all economies and the corresponding bidder names (as strings) of the bidders which are active in these economies.

        self.new_query_RS_STATS = {}
        self.new_query_DE_STATS = {}
        self.sampled_marginals_per_iteration = OrderedDict()
        self.efficiency_per_iteration = OrderedDict()  # storage for efficiency stat per auction round
        self.efficient_allocation_per_iteration = OrderedDict()  # storage for efficent allocation per auction round
        self.efficiency_per_iteration_all_economies = OrderedDict()  # storage for efficeny stat per auction round for all economies (including marginals)
        self.nn_seed = self.SATS_auction_instance_seed * 10 ** 4  # seed for training the MVNNs, shifted by 10**6 (i.e. one has 10**6 fits capacity until its overlapping with self.nn_seed+1 )
        self.mip_logs = defaultdict(list)
        self.train_logs = {}
        self.total_time_elapsed = None
        self.ml_total_train_time_sec = 0  # sums together train_logs thus does not account for parallelization
        self.total_time_elapsed_estimation_step = 0

        # DYNAMIC PER ECONOMY
        self.economy_status = OrderedDict(list((key, False) for key, value in
                                               self.economies.items()))  # boolean, status of economy: if already calculated.
        self.mlca_marginal_allocations = OrderedDict(list((key, None) for key, value in self.economies.items() if
                                                          key != 'Main Economy'))  # Allocation of the WDP based on the elicited bids
        self.mlca_marginal_scws = OrderedDict(list((key, None) for key, value in self.economies.items() if
                                                   key != 'Main Economy'))  # Social Welfare of the Allocation of the WDP based on the elicited bids

        self.mlca_efficiency_marginals = OrderedDict(list((key, None) for key, value in self.economies.items() if
                                                          key != 'Main Economy'))  # Social Welfare of the Allocation of the WDP based on the elicited bids

        self.elapsed_time_mip = OrderedDict(
            list((key, []) for key, value in self.economies.items()))  # stored MIP solving times per economy
        self.warm_start_sol = OrderedDict(list((key, None) for key, value in
                                               self.economies.items()))  # MIP SolveSolution object used as warm start per economy
        self.sampled_marginals_counter = OrderedDict(
            list((key, 0) for key, value in self.economies.items() if key != 'Main Economy'))

        self.RS_allocation_cache_for_reusing = OrderedDict(list((key, []) for key, value in self.economies.items()))
        self.DE_allocation_cache_for_reusing = OrderedDict(list((key, []) for key, value in self.economies.items()))

        # DYNAMIC PER BIDDER
        self.mlca_payments = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in
                                              self.bidder_ids))  # VCG-style payments in MLCA, calculated at the end
        self.elicited_bids = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in
                                              self.bidder_ids))  # R=(R_1,...,R_n) elicited bids per bidder
        self.current_query_profile = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in
                                                      self.bidder_ids))  # S=(S_1,...,S_n) number of actual value queries, that takes into account if the same bundle was queried from a bidder in two different economies that it is not counted twice
        self.NN_parameters = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in
                                              self.bidder_ids))  # DNNs parameters as in the Class NN described.

        # DYNAMIC PER ECONOMY & BIDDER
        self.argmax_allocation = OrderedDict(
            list((key, OrderedDict(list((bidder_id, [None, None]) for bidder_id in value))) for key, value in
                 self.economies_names.items()))  # [a,a_restr] a: argmax bundles per bidder and economy, a_restr: restricted argmax bundles per bidder and economy

        self.sampled_marginals_bidder_counter = OrderedDict(list((bidder_id,
                                                                  OrderedDict(list(
                                                                      (key, 0) for key in self.economies_names if
                                                                      key not in ['Main Economy',
                                                                                  f'Marginal Economy -({bidder_id})']))
                                                                  ) for bidder_id in self.bidder_names))

        if self.separate_economy_training:
            self.NN_models = OrderedDict(list(
                (key, OrderedDict(list((bidder_id, None) for bidder_id in value))) for key, value in
                self.economies_names.items()))
        else:
            self.NN_models = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in self.bidder_ids))

    def save_results(self,
                     res_path
                     ):

        save_dict = OrderedDict()

        save_dict['MLCA Efficiency'] = self.mlca_allocation_efficiency
        save_dict[
            'MLCA Efficiency Marginals'] = self.mlca_efficiency_marginals  # Potentially None for intermediate iteration
        save_dict['MLCA Efficiency per Iteration'] = self.efficiency_per_iteration
        save_dict[
            'MLCA Efficiency per Iteration all Economies'] = self.efficiency_per_iteration_all_economies  # only for new_query_option= 'RS' or 'DE'

        save_dict['MLCA Optimal SCW Main'] = self.mlca_scw,  # Potentially None for intermediate iteration
        save_dict[
            'MLCA Optimal SCW Marginals'] = self.mlca_marginal_scws,  # Potentially None for intermediate iteration
        save_dict['SATS Optimal SCW'] = self.SATS_auction_instance_scw

        save_dict['MLCA Relative Revenue'] = self.relative_revenue,  # None for intermediate iteration
        save_dict['MLCA Payments'] = self.mlca_payments,  # None for intermediate iteration

        save_dict['Total Time Elapsed'] = self.total_time_elapsed
        save_dict['Total Time Elapsed Distribution'] = self.total_time_elapsed_distr

        save_dict['MIP Statistics'] = {'Elapsed Times of MIPs': self.elapsed_time_mip,
                                       'Optimizations': self.number_of_optimization,
                                       'mip_statistics': self.mip_logs,
                                       }

        save_dict['MLCA Sampled Marginals per Iteration'] = self.sampled_marginals_per_iteration
        save_dict['MLCA Sampled Marginals Counter'] = self.sampled_marginals_counter
        save_dict['MLCA Sampled Marginals Bidder Counter'] = self.sampled_marginals_bidder_counter

        save_dict['RS New query Statistics'] = self.new_query_RS_STATS
        save_dict['DE New query Statistics'] = self.new_query_DE_STATS

        save_dict['ML Statistics'] = self.train_logs

        save_dict['MLCA Allocation'] = self.mlca_allocation,  # Potentially None for intermediate iteration
        save_dict['SATS Efficient Allocation'] = self.SATS_auction_instance_allocation
        save_dict['MLCA Allocation per Iteration'] = self.efficient_allocation_per_iteration

        save_dict['MLCA Elicited Bids'] = self.elicited_bids

        json.dump(save_dict, open(os.path.join(res_path, 'results.json'), 'w'), indent=4,
                  sort_keys=False, separators=(', ', ': '),
                  ensure_ascii=False, cls=NumpyEncoder)

    def get_info(self,
                 final_summary=False
                 ):

        if final_summary:
            logging.warning('SUMMARY')
        else:
            logging.warning('INFO')
        logging.warning('-----------------------------------------------')
        logging.warning('Seed Auction Instance: %s', self.SATS_auction_instance_seed)
        logging.warning('Iteration of MLCA: %s', self.mlca_iteration)
        logging.warning('Number of Elicited Bids:')
        for k, v in self.get_number_of_elicited_bids().items():
            logging.warning(k + ': %s', v)
        logging.warning('Qinit: %s | Qround: %s | Qmax: %s', self.Qinit, self.Qround, self.Qmax)
        if final_summary:
            logging.warning('EFFICIENCY: {} %'.format(round(self.mlca_allocation_efficiency, 4) * 100))
            logging.warning(f'TOTAL TIME ELAPSED: {self.total_time_elapsed[0]}')
            logging.warning(
                f'MIP TIME: {self.total_time_elapsed_distr["MIP"][0]} ({100 * self.total_time_elapsed_distr["MIP"][1]}%) | ML TRAIN TIME: {self.total_time_elapsed_distr["ML_TRAIN"][0]} ({100 * self.total_time_elapsed_distr["ML_TRAIN"][1]}%) | OTHER TIME: {self.total_time_elapsed_distr["OTHER"][0]} ({100 * self.total_time_elapsed_distr["OTHER"][1]}%)')
            if self.new_query_option == 'RS':
                logging.warning(
                    f'NORMAL OPTIMIZATIONS: {self.number_of_optimization["normal"]}/ BIDDER SPECIFIC MIP: {self.number_of_optimization["bidder_specific_MIP"]} /BIDDER SPECIFIC RS: {self.number_of_optimization["bidder_specific_RS"]} /BIDDER SPECIFIC RS REUSED: {self.number_of_optimization["bidder_specific_RS_REUSED"]}')
            elif self.new_query_option == 'DE':
                logging.warning(
                    f'NORMAL OPTIMIZATIONS: {self.number_of_optimization["normal"]}/ BIDDER SPECIFIC MIP: {self.number_of_optimization["bidder_specific_MIP"]} /BIDDER SPECIFIC DE: {self.number_of_optimization["bidder_specific_DE"]} /BIDDER SPECIFIC DE REUSED: {self.number_of_optimization["bidder_specific_DE_REUSED"]}')
            else:
                logging.warning(
                    f'NORMAL OPTIMIZATIONS: {self.number_of_optimization["normal"]}/ BIDDER SPECIFIC MIP: {self.number_of_optimization["bidder_specific_MIP"]}')
            logging.warning(
                f'MIP avg REL.GAP: {np.mean(self.mip_logs["rel. gap"])} | MIP HIT TIME LIMIT: {int(sum(self.mip_logs["hit_limit"]))}/{len(self.mip_logs["hit_limit"])}')
            logging.warning('MLCA FINISHED')
        else:
            logging.warning('Efficiency given elicited bids from iteration 0-%s: %s\n',
                            self.mlca_iteration - 1,
                            self.efficiency_per_iteration.get(self.mlca_iteration - 1))

    def calc_time_spent(self,
                        ):

        self.end_time = datetime.now()
        if self.start_time:
            tot_seconds = (self.end_time - self.start_time).total_seconds()
            mip_seconds = sum(self.mip_logs['time'])
            ml_estimation_seconds = self.total_time_elapsed_estimation_step
            other_seconds = tot_seconds - mip_seconds - ml_estimation_seconds
            self.total_time_elapsed_distr['MIP'] = (
                '{}d {}h:{}m:{}s'.format(*timediff_d_h_m_s(timedelta(seconds=mip_seconds))),
                round(mip_seconds / tot_seconds, ndigits=2),
                mip_seconds)
            self.total_time_elapsed_distr['ML_TRAIN'] = (
                '{}d {}h:{}m:{}s'.format(*timediff_d_h_m_s(timedelta(seconds=ml_estimation_seconds))),
                round(ml_estimation_seconds / tot_seconds, ndigits=2),
                ml_estimation_seconds)
            self.total_time_elapsed_distr['OTHER'] = (
                '{}d {}h:{}m:{}s'.format(*timediff_d_h_m_s(timedelta(seconds=other_seconds))),
                round(other_seconds / tot_seconds, ndigits=2),
                other_seconds)
            self.total_time_elapsed = ('{}d {}h:{}m:{}s'.format(*timediff_d_h_m_s(self.end_time - self.start_time)),
                                       tot_seconds)
        else:
            logging.warning('start_time not specififed')

    def print_argmax_allocation(self,
                                economy_key):
        logging.info('')
        logging.info(f'current ARGMAX ALLOCATION | BIDDER SPECIFIC BUNDLES in {economy_key}:')
        logging.info('-----------------------------------------------')
        for key, value in self.argmax_allocation[economy_key].items():
            logging.info(key + ':  %s | %s', value[0], value[1])

    def get_number_of_elicited_bids(self,
                                    bidder=None
                                    ):

        if bidder is None:
            return OrderedDict((bidder, self.elicited_bids[bidder][0].shape[0] - 1) for bidder in
                               self.bidder_names)  # -1 since empty-bundle is included
        else:
            return self.elicited_bids[bidder][0].shape[0] - 1  # -1 since empty-bundle is included

    def calculate_efficiency_per_iteration(self):
        logging.debug('')
        logging.debug('Calculate current efficiency:')
        allocation, objective = self.solve_WDP(self.elicited_bids, verbose=1)
        self.efficient_allocation_per_iteration[self.mlca_iteration] = allocation
        efficiency = self.calculate_efficiency_of_allocation(allocation=allocation, allocation_scw=objective)
        logging.debug('Current efficiency: {}'.format(efficiency))
        self.efficiency_per_iteration[self.mlca_iteration] = efficiency
        return efficiency

    def set_calc_efficiency_per_iteration_all_economies(self,
                                                        mlca_iteration: int
                                                        ):

        '''
        Sets calculated SCWs of marginals & main (only for new_query_option = 'RS' or 'DE') in logs
        '''

        if self.new_query_option == 'RS' or self.new_query_option == 'DE':
            self.efficiency_per_iteration_all_economies[mlca_iteration] = {}
            # MARGINALS ECONOMIES
            for k, marginal_scw in self.mlca_marginal_scws.items():
                if marginal_scw:
                    self.efficiency_per_iteration_all_economies[mlca_iteration][
                        k] = marginal_scw / self.SATS_auction_instance_scw
                else:
                    self.efficiency_per_iteration_all_economies[mlca_iteration][k] = None
            # MAIN ECONOMY
            if self.mlca_scw:
                self.efficiency_per_iteration_all_economies[mlca_iteration][
                    'Main Economy'] = self.mlca_scw / self.SATS_auction_instance_scw
            else:
                self.efficiency_per_iteration_all_economies[mlca_iteration]['Main Economy'] = None

    def set_NN_parameters(self,
                          parameters
                          ):

        logging.debug('Set NN parameters')
        self.NN_parameters = OrderedDict(parameters)

    def set_MIP_parameters(self,
                           parameters
                           ):

        logging.debug('Set MIP parameters')
        self.MIP_parameters = parameters

    def set_DE_parameters(self,
                          parameters
                          ):

        logging.debug('Set DE parameters')
        self.DE_parameters = parameters

    def set_RS_parameters(self,
                          parameters
                          ):

        logging.debug('Set DE parameters')
        self.RS_parameters = parameters

    def set_initial_bids(self,
                         initial_bids=None,
                         fitted_scaler=None,
                         seed=None,
                         include_full_bundle=True):

        logging.info('INITIALIZE BIDS')
        logging.info('-----------------------------------------------')
        if initial_bids is None:  # Uniform sampling (now with correct isLegacy Version)
            self.elicited_bids, self.fitted_scaler = initial_bids_mlca_unif(
                SATS_auction_instance=self.SATS_auction_instance,
                number_initial_bids=self.Qinit, bidder_names=self.bidder_names,
                scaler=self.scaler, seed=seed, include_full_bundle=include_full_bundle)
        else:
            assert list(initial_bids.keys()) == self.bidder_names
            logging.debug('Setting inputed initial bundle-value pairs of dimenions:')
            for k, v in initial_bids.items():
                logging.debug(k + ': X=%s, Y=%s', v[0].shape, v[1].shape)
            self.elicited_bids = initial_bids  # needed format: {Bidder_i:[bundles,values]}
            self.fitted_scaler = fitted_scaler  # fitted scaler to initial bids
            self.Qinit = [v[0].shape[0] for k, v in initial_bids.items()]

    def reset_argmax_allocations(self):
        self.argmax_allocation = OrderedDict(list(
            (key, OrderedDict(list((bidder_id, [None, None]) for bidder_id in value))) for key, value in
            self.economies_names.items()))

    def reset_allocation_cache_for_reusing(self):
        self.RS_allocation_cache_for_reusing = OrderedDict(list((key, []) for key, value in self.economies.items()))
        self.DE_allocation_cache_for_reusing = OrderedDict(list((key, []) for key, value in self.economies.items()))

    def reset_current_query_profile(self):
        self.current_query_profile = OrderedDict(
            list(('Bidder_{}'.format(bidder_id), None) for bidder_id in self.bidder_ids))

    def reset_NN_models(self):
        delattr(self, 'NN_models')
        self.NN_models = OrderedDict(list(
            (key, OrderedDict(list((bidder_id, None) for bidder_id in value))) for key, value in
            self.economies_names.items()))

    def reset_economy_status(self):
        self.economy_status = OrderedDict(list((key, False) for key, value in self.economies.items()))

    def reset_scws_and_allocations(self):

        '''
        Resets calculated SCWs of marginals & main (were calculated only for new_query_option = 'RS' or 'DE')
        '''

        # marginals
        self.mlca_marginal_allocations = OrderedDict(
            list((key, None) for key, value in self.economies.items() if key != 'Main Economy'))
        self.mlca_marginal_scws = OrderedDict(
            list((key, None) for key, value in self.economies.items() if key != 'Main Economy'))
        self.mlca_efficiency_marginals = OrderedDict(
            list((key, None) for key, value in self.economies.items() if key != 'Main Economy'))
        # main
        self.mlca_allocation = None
        self.mlca_scw = None

    def solve_SATS_auction_instance(self):
        self.SATS_auction_instance_allocation, self.SATS_auction_instance_scw = self.SATS_auction_instance.get_efficient_allocation()

    def set_global_admissible_marginals(self,
                                        mlca_iteration
                                        ):

        '''
        Sets global Qround many marginals per iteration in a balanced deterministic way.
        '''

        # Random shuffling
        l = list(self.sampled_marginals_counter.items())
        random.shuffle(l)
        d_shuffled = dict(l)

        admissible_marginals = [k for k, v in sorted(d_shuffled.items(), key=lambda item: item[1])][:self.Qround]

        # update sampled marginals and their respective counter
        self.sampled_marginals_per_iteration[self.mlca_iteration] = []
        for k in admissible_marginals:
            self.sampled_marginals_counter[k] += 1
            self.sampled_marginals_per_iteration[self.mlca_iteration].append(k)

    def sample_marginal_economies_for_bidder(self,
                                             active_bidder
                                             ):

        '''
        Sets per iteration for a single bidder her admissible marginals.
        Remark: If Qround=1 then return immediately since there are no marginal queries
        '''

        if self.Qround == 1:
            self.sampled_marginals_per_iteration[self.mlca_iteration] = []
            return []

        # marginals sampled globally for all bidders in each iteration (frequency balanced is hard coded)
        if self.balanced_global_marginals:
            if not self.sampled_marginals_per_iteration.get(self.mlca_iteration):
                self.set_global_admissible_marginals(mlca_iteration=self.mlca_iteration)

            admissible_marginals_active_bidder = [m for m in self.sampled_marginals_per_iteration[self.mlca_iteration]
                                                  if m != f'Marginal Economy -({key_to_int(active_bidder)})']
            # no choice -> bidder gets all the admissible_marginals_active_bidder
            if len(admissible_marginals_active_bidder) == (self.Qround - 1):
                sampled_marginals_active_bidder = admissible_marginals_active_bidder
            # choice -> select our of the Qround admissible ones the ones which have been selected the least often for active bidder
            elif len(admissible_marginals_active_bidder) == self.Qround:
                helper_dict = {k: self.sampled_marginals_bidder_counter[active_bidder][k] for k in
                               admissible_marginals_active_bidder}
                # Random shuffling
                l = list(helper_dict.items())
                random.shuffle(l)
                d_shuffled = dict(l)

                sampled_marginals_active_bidder = [k for k, v in sorted(d_shuffled.items(), key=lambda item: item[1])][
                                                  :(self.Qround - 1)]
            else:
                raise ValueError(
                    f'Invalid admissible sampled marginals: {admissible_marginals_active_bidder} for {active_bidder} and Qround:{self.Qround}')
        # marginals sampled individually for each bidder in each iteration
        else:
            admissible_marginals_active_bidder = [x for x in list((self.economies.keys())) if x not in ['Main Economy',
                                                                                                        f'Marginal Economy -({key_to_int(active_bidder)})']]
            sampled_marginals_active_bidder = random.sample(admissible_marginals_active_bidder, k=self.Qround - 1)

        for k in sampled_marginals_active_bidder:
            self.sampled_marginals_bidder_counter[active_bidder][k] += 1

        return sampled_marginals_active_bidder

    def update_elicited_bids(self):

        for bidder in self.bidder_names:
            logging.info('UPDATE ELICITED BIDS: S -> R for %s', bidder)
            logging.info('---------------------------------------------')
            # update bundles
            self.elicited_bids[bidder][0] = np.append(self.elicited_bids[bidder][0],
                                                      self.current_query_profile[bidder],
                                                      axis=0)
            # update values
            bidder_value_reports = self.value_queries(bidder_id=key_to_int(bidder),
                                                      bundles=self.current_query_profile[bidder])
            self.elicited_bids[bidder][1] = np.append(self.elicited_bids[bidder][1],
                                                      bidder_value_reports,
                                                      axis=0)

            logging.info('CHECK Uniqueness of updated elicited bids:')
            check = len(np.unique(self.elicited_bids[bidder][0], axis=0)) == len(self.elicited_bids[bidder][0])
            if check:
                logging.info('UNIQUE\n')
            else:
                raise RuntimeError('NOT UNIQUE\n')
        return (check)

    def update_current_query_profile(self,
                                     bidder,
                                     bundle_to_add
                                     ):

        '''
        Adds bundle bundle_to_add to current query profile S
        Remark: bundle_to_add is expected to be a new bundle -> error will  be thrown if invalid dim or alrteady elicited
        '''

        # DIM Check
        if bundle_to_add.shape != (self.M,):
            raise RuntimeError('No valid bundle dim -> CANNOT ADD BUNDLE.')
        # NEW CHECK
        if self.check_bundle_contained(bundle=bundle_to_add, bidder=bidder):
            raise RuntimeError('Bundle already elicited -> CANNOT ADD BUNDLE.')
        logging.info('')
        logging.info('ADD BUNDLE to current query profile S')
        if self.current_query_profile[bidder] is None:
            self.current_query_profile[bidder] = bundle_to_add.reshape(1, -1)
        else:
            self.current_query_profile[bidder] = np.append(self.current_query_profile[bidder],
                                                           bundle_to_add.reshape(1, -1),
                                                           axis=0)

    def value_queries(self,
                      bidder_id,
                      bundles,
                      return_on_original_scale=False
                      ):
        '''
        REMARK: value queries always rescaled to true values
        '''

        raw_values = np.array(
            [self.SATS_auction_instance.calculate_value(bidder_id, bundles[k, :]) for k in range(bundles.shape[0])])
        if not self.fitted_scaler or return_on_original_scale:
            logging.debug('Return raw value queries')
            return (raw_values)
        else:
            minI = int(round(self.fitted_scaler.data_min_[0] * self.fitted_scaler.scale_[0]))
            maxI = int(round(self.fitted_scaler.data_max_[0] * self.fitted_scaler.scale_[0]))
            logging.debug('*SCALING*')
            logging.debug('---------------------------------------------')
            logging.debug('raw values %s', raw_values)
            logging.debug('Return value queries scaled by: %s to the interval [%s,%s]',
                          round(self.fitted_scaler.scale_[0], 8), minI, maxI)
            logging.debug('scaled values %s', self.fitted_scaler.transform(raw_values.reshape(-1, 1)).flatten())
            logging.debug('---------------------------------------------')

            return (self.fitted_scaler.transform(raw_values.reshape(-1, 1)).flatten())

    def check_bundle_contained(self,
                               bundle,
                               bidder,
                               log_level='INFO'):

        if np.any(np.equal(self.elicited_bids[bidder][0], bundle).all(axis=1)):
            if log_level == 'INFO':
                logging.info('Bundle ALREADY ELICITED from {}'.format(bidder))
            elif log_level == 'DEBUG':
                logging.debug('Bundle ALREADY ELICITED from {}'.format(bidder))
            elif log_level == 'WARNING':
                logging.warning('Bundle ALREADY ELICITED from {}'.format(bidder))
            else:
                pass
            return (True)
        if self.current_query_profile[bidder] is not None:
            if np.any(np.equal(self.current_query_profile[bidder], bundle).all(axis=1)):
                if log_level == 'INFO':
                    logging.info('Bundle ALREADY QUERIED IN THIS ITERATION from {}'.format(bidder))
                elif log_level == 'DEBUG':
                    logging.debug('Bundle ALREADY QUERIED IN THIS ITERATION from {}'.format(bidder))
                elif log_level == 'WARNING':
                    logging.warning('Bundle ALREADY QUERIED IN THIS ITERATION from {}'.format(bidder))
                else:
                    pass
                return (True)
        return (False)

    def check_exp_100_uUB_SCW_is_larger(self,
                                        allocation,
                                        economy_key,
                                        precalculated_exp_100_uUB_scw=None
                                        ):

        # CHECK 2: IF 100% SCW OF PROPOSED RANDOM ALLOCATION IS LARGER OR EQUAL TO CURRENT EFFICIENCY IN ECONOMY
        # --------------------------------------------------------------
        if precalculated_exp_100_uUB_scw is None:
            # Remark: return on original scale such that comparable to self.mlca_marginal_scws[economy_key], which is not scaled
            exp_100_uUB_scw = self.calculate_exp_100_uUB_scw(allocation=allocation, economy_key=economy_key)[0]
        else:
            exp_100_uUB_scw = precalculated_exp_100_uUB_scw

        logging.debug(f'EXP_100_uUB_SCW OF PROPOSED ALLOCATION: {exp_100_uUB_scw}')

        current_economy_scw = self.get_current_economy_scw(economy_key)

        if exp_100_uUB_scw < current_economy_scw:
            logging.debug('100% SCW OF PROPOSED ALLOCATION IS SMALLER THAN CURRENT SCW IN THIS ECONOMY')
            return False, exp_100_uUB_scw, current_economy_scw
        else:
            return True, exp_100_uUB_scw, current_economy_scw
        # --------------------------------------------------------------

    # def check_exp_100_uUB_SCW_is_larger_population(self,
    #                                     population,
    #                                     economy_key,
    #                                     precalculated_exp_100_uUB_scw=None
    #                                     ):

    #     # CHECK 2: IF 100% SCW OF PROPOSED RANDOM ALLOCATION IS LARGER OR EQUAL TO CURRENT EFFICIENCY IN ECONOMY
    #     # --------------------------------------------------------------
    #     if precalculated_exp_100_uUB_scw is None:
    #         # Remark: return on original scale such that comparable to self.mlca_marginal_scws[economy_key], which is not scaled
    #         exp_100_uUB_scw = self.calculate_exp_100_uUB_scw(allocation=population, economy_key=economy_key)
    #         #Question: Why was there "[0]" at the end of the line?
    #     else:
    #         exp_100_uUB_scw = precalculated_exp_100_uUB_scw

    #     logging.debug(f'EXP_100_uUB_SCW OF PROPOSED ALLOCATION: {exp_100_uUB_scw}')
    #     #Question: Can I print a numpy array this way?

    #     current_economy_scw = self.get_current_economy_scw(economy_key)

    #     return exp_100_uUB_scw < current_economy_scw, exp_100_uUB_scw, current_economy_scw

    #     #QUestion: Is it a problem that we don't log the True/False values?
    #     # if exp_100_uUB_scw < current_economy_scw:
    #     #     logging.debug('100% SCW OF PROPOSED ALLOCATION IS SMALLER THAN CURRENT SCW IN THIS ECONOMY')
    #     #     return False, exp_100_uUB_scw, current_economy_scw
    #     # else:
    #     #     return True, exp_100_uUB_scw, current_economy_scw
    #     # --------------------------------------------------------------

    def get_current_economy_scw(self,
                                economy_key):

        if economy_key == 'Main Economy':
            # Calculate SCW from elicited bids if not already calculated
            if not self.mlca_scw:
                self.calculate_mlca_allocation(economy=economy_key)
            current_economy_scw = self.mlca_scw
        else:
            # Calculate SCW from elicited bids if not already calculated
            if not self.mlca_marginal_scws[economy_key]:
                self.calculate_mlca_allocation(economy=economy_key)
            current_economy_scw = self.mlca_marginal_scws[economy_key]
        logging.debug(f'CURRENT MARGINAL SCW: {current_economy_scw}')
        return current_economy_scw

    def calculate_exp_100_uUB_scw(self,
                                  allocation,
                                  economy_key,
                                  ):
        '''
        Remark: allocation is a dict: {bidder_name: x} where x is a np.array \in {0,1}^m
        Remark: returns scw always on original scale
        '''

        MODELS = self.get_ML_models(economy_key, model_type='exp_100_uUB_model')
        assert list(allocation.keys()) == list(MODELS.keys())

        scw = 0
        for bidder, bundle in allocation.items():
            MODELS[bidder].eval()
            if type(bundle) == list:
                bundle = np.array(bundle)
            v = MODELS[bidder](torch.from_numpy(bundle).float()).detach().numpy()

            v *= MODELS[bidder]._target_max / self.local_scaling_factor  # revert local scaling

            if self.fitted_scaler:
                v = self.fitted_scaler.inverse_transform(v)  # revert global scaling
            scw += v

        return scw.reshape(-1)

    def calculate_uUB_scw(self, allocation, economy_key):
        '''
        Remark: allocation is a dict: {bidder_name: x} where x is a np.array \in {0,1}^m
        Remark: returns scw always on original scale
        '''

        MODELS = self.get_ML_models(economy_key, model_type='uUB_model')
        assert list(allocation.keys()) == list(MODELS.keys())

        scw = 0
        for bidder, bundle in allocation.items():
            if bidder in MODELS:
                MODELS[bidder].eval()
                if type(bundle) == list:
                    bundle = np.array(bundle)
                v = MODELS[bidder](torch.from_numpy(bundle).float()).detach().numpy()

                v *= MODELS[bidder]._target_max / self.local_scaling_factor  # revert local scaling

                if self.fitted_scaler:
                    v = self.fitted_scaler.inverse_transform(v)  # revert global scaling
                scw += v

        return scw.reshape(-1)

    def create_population_check1(self,
                                 desired_population_size,
                                 max_init_attempts,
                                 economy_key,
                                 active_bidder,
                                 ):
        '''
        Create a new population for the current marginal economy that fulfills check 1, not necessarily check 2.
        '''
        population = defaultdict(list)
        cur_population_size = 0
        num_attempts = 0
        while cur_population_size < desired_population_size and num_attempts < max_init_attempts:
            num_attempts += 1
            random_allocation = self.get_random_allocation(economy_key=economy_key)

            # CHECK 1: IF ACTIVE BIDDER BUNDLE HAS BEEN ALREADY ELICITED
            # --------------------------------------------------------------
            if self.check_bundle_contained(bundle=random_allocation[active_bidder],
                                           bidder=active_bidder,
                                           log_level='DEBUG'):
                self.new_query_DE_STATS[self.mlca_iteration][economy_key][active_bidder]['failed_CHECK1_init_pop'] += 1
                continue
            # --------------------------------------------------------------

            # SUCCESSFUL:
            # --------------------------------------------------------------
            cur_population_size += 1
            for bidder, v in random_allocation.items():
                population[bidder].append(v.flatten())
            # --------------------------------------------------------------

        logging.info(f'Reached population check1 size: {cur_population_size}')
        if cur_population_size == desired_population_size:
            logging.info('population check1 sucessfully created.')
            population_size = desired_population_size
            return population, population_size

        elif cur_population_size > desired_population_size / 2:
            # This case should be extremly rare
            logging.info('population check1 unsuccessfull, but reached at least desired_population_size/2.')
            population_size = cur_population_size
            return population, population_size

        else:
            logging.info(
                'population check1 unsuccessfull, solving the restricted MIP? No plan what is happening in this case')
            # TODO This case should be avoided. Test, waht happens in this case
            return None, None

    def create_initial_DE_population(self,
                                     desired_population_size,
                                     max_init_attempts,
                                     economy_key,
                                     active_bidder,
                                     ):

        # Create a new population for the current marginal economy
        population = defaultdict(list)
        cur_population_size = 0
        num_batches = 0
        num_eval = 0
        batching_factor = 1.1
        current_economy_scw = self.get_current_economy_scw(economy_key)
        while cur_population_size < desired_population_size and num_eval < max_init_attempts:
            num_batches += 1
            remaining_eval_budget = max_init_attempts - num_eval
            if remaining_eval_budget < desired_population_size * 0.5 - cur_population_size:
                logging.info('Initial DE population unsuccessfull, solving the restricted MIP.')
                return None, None
            eval_batch_size = max(64, min(remaining_eval_budget,
                                          int((desired_population_size - cur_population_size) * batching_factor) + 1))
            population_check1, population_check1_size = self.create_population_check1(
                desired_population_size=eval_batch_size,
                max_init_attempts=desired_population_size * batching_factor * 100,
                economy_key=economy_key,
                active_bidder=active_bidder,
            )
            exp_100_uUB_scw = self.calculate_exp_100_uUB_scw(allocation=population_check1, economy_key=economy_key)
            for i in range(population_check1_size):
                # # CHECK 2: IF 100% SCW OF PROPOSED RANDOM ALLOCATION IS LARGER OR EQUAL TO CURRENT EFFICIENCY IN ECONOMY
                # # --------------------------------------------------------------
                if exp_100_uUB_scw[i] < current_economy_scw:
                    logging.debug('100% SCW OF PROPOSED ALLOCATION IS SMALLER THAN CURRENT SCW IN THIS ECONOMY')
                    self.new_query_DE_STATS[self.mlca_iteration][economy_key][active_bidder][
                        'failed_CHECK2_init_pop'] += 1
                    # return False, exp_100_uUB_scw, current_economy_scw
                # --------------------------------------------------------------
                else:
                    # return True, exp_100_uUB_scw, current_economy_scw
                    # SUCCESSFUL:
                    # --------------------------------------------------------------
                    cur_population_size += 1
                    for bidder in self.economies_names[economy_key]:
                        population[bidder].append(population_check1[bidder][i])
                    # --------------------------------------------------------------
            num_eval += population_check1_size
            success_rate_check_2 = cur_population_size / num_eval
            if num_batches == 1 and success_rate_check_2 * max_init_attempts < 0.6 * desired_population_size:
                logging.info('Initial DE population unsuccessfull, solving the restricted MIP.')
                return None, None
            batching_factor = 1 / success_rate_check_2
            # # CHECK 2: IF 100% SCW OF PROPOSED RANDOM ALLOCATION IS LARGER OR EQUAL TO CURRENT EFFICIENCY IN ECONOMY
            # # --------------------------------------------------------------
            # if not self.check_exp_100_uUB_SCW_is_larger(allocation=random_allocation,
            #                                             economy_key=economy_key,
            #                                             precalculated_exp_100_uUB_scw=None)[0]:
            #     self.new_query_DE_STATS[self.mlca_iteration][economy_key][active_bidder]['failed_CHECK2_init_pop'] += 1
            #     continue
            # # --------------------------------------------------------------
            # CHECK 2: IF 100% SCW OF PROPOSED RANDOM ALLOCATION IS LARGER OR EQUAL TO CURRENT EFFICIENCY IN ECONOMY
            # --------------------------------------------------------------
            # check2_bools self.check_exp_100_uUB_SCW_is_larger(allocation=population_check1,
            #                                             economy_key=economy_key,
            #                                             precalculated_exp_100_uUB_scw=exp_100_uUB_scw): #WHy "[0]"?
            #     self.new_query_DE_STATS[self.mlca_iteration][economy_key][active_bidder]['failed_CHECK2_init_pop'] += 1
            #     continue
            # # --------------------------------------------------------------

            # # SUCCESSFUL:
            # # --------------------------------------------------------------
            # cur_population_size += 1
            # for bidder, v in random_allocation.items():
            #     population[bidder].append(v.flatten())
            # # --------------------------------------------------------------

        logging.info(f'Reached population size: {cur_population_size}')
        if cur_population_size >= desired_population_size:
            logging.info('Initial DE population sucessfully created.')
            population_size = cur_population_size
            return population, population_size

        elif cur_population_size > desired_population_size / 2:
            logging.info('Initial DE population unsuccessfull, but reached at least desired_population_size/2.')
            population_size = cur_population_size
            return population, population_size

        else:
            logging.info('Initial DE population unsuccessfull, solving the restricted MIP.')
            return None, None

    def get_new_DE_bundle(self,
                          economy_key,
                          active_bidder,
                          iterations=10000):

        logging.info('')
        logging.info(f'GENERATE NEW DE BUNDLE in {economy_key} for {active_bidder}:')
        logging.info('-----------------------------------------------')
        # Initialize new_random_exp_100_uUB_bundle save statistics
        if not self.new_query_DE_STATS.get(self.mlca_iteration):
            self.new_query_DE_STATS[self.mlca_iteration] = {}
        if not self.new_query_DE_STATS[self.mlca_iteration].get(economy_key):
            self.new_query_DE_STATS[self.mlca_iteration][economy_key] = {}
        if not self.new_query_DE_STATS[self.mlca_iteration][economy_key].get(active_bidder):
            self.new_query_DE_STATS[self.mlca_iteration][economy_key][active_bidder] = {
                'failed_init_pop': False,
                'failed_CHECK1_init_pop': 0,
                'failed_CHECK2_init_pop': 0,
                'init_pop_size': None,
                'failed_CHECK1': 0,
                'failed_CHECK2': 0,
                'failed_CHECK_SCW': 0,
                'iterations': 0,
                'true_scw': None,
                'incumbent_eval_scw': None,
                'reused': False,
                'incumbent_trajectory': {}
            }

        # REUSE 1: Check if we can use bundle from random_allocation generated for a different (previous) bidder in this economy
        for prev_allocation in self.DE_allocation_cache_for_reusing[economy_key]:
            logging.debug(f'DE ALLOC FROM {economy_key}:{prev_allocation}')
            logging.debug(f'{active_bidder} BUNDLE:{prev_allocation[active_bidder].reshape(-1, )}')
            # CHECK 1 (only needed here since CHECK 2 has already been performed): IF ACTIVE BIDDER BUNDLE HAS BEEN ALREADY ELICITED
            # --------------------------------------------------------------
            if not self.check_bundle_contained(bundle=prev_allocation[active_bidder], bidder=active_bidder,
                                               log_level='DEBUG'):
                logging.info('REUSE BUNDLE FROM A PREVIOUS DE ALLOC IN THIS ECONOMY AND ITERATION')
                self.new_query_DE_STATS[self.mlca_iteration][economy_key][active_bidder]['reused'] = True
                logging.info('STATISTICS:')
                for k, v in self.new_query_DE_STATS[self.mlca_iteration][economy_key][
                    active_bidder].items():
                    logging.info(f'->{k}:{v}')
                self.number_of_optimization["bidder_specific_DE_REUSED"] += 1
                return prev_allocation[active_bidder].reshape(-1, ).astype(int)
            # --------------------------------------------------------------

        def convert_to_item_bidder_matrix(individual_idx: int, population: dict, num_items: int):
            item_bidder_matrix = np.zeros(shape=(len(population), num_items))
            for bidder_idx, bidder in enumerate(sorted(population.keys())):  # Todo: Check sorting
                bundle = population[bidder][individual_idx]  # 1hot encoded [num_items]
                item_bidder_matrix[bidder_idx, :] = bundle
            return item_bidder_matrix

        # DE Params
        desired_population_size = self.DE_parameters['desired_population_size']
        mutation_strength = self.DE_parameters['mutation_strength']
        crossover_prob = self.DE_parameters['crossover_prob']
        mutation_strategy = self.DE_parameters['mutation_strategy']
        reprojection_strategy = self.DE_parameters['reprojection_strategy']
        crossover_strategy = self.DE_parameters['crossover_strategy']
        softmax_temperature = self.DE_parameters['softmax_temperature']
        patience_de_iterations = self.DE_parameters['patience_de_iterations']
        max_init_attempts = self.DE_parameters['max_init_attempts']

        # DE Initial Population
        population, population_size = self.create_initial_DE_population(desired_population_size=desired_population_size,
                                                                        max_init_attempts=max_init_attempts,
                                                                        economy_key=economy_key,
                                                                        active_bidder=active_bidder)
        if population is None:
            # Remark: solves then restricted MIP in next_queries Option 4
            self.new_query_DE_STATS[self.mlca_iteration][economy_key][active_bidder]['failed_init_pop'] = True
            return None

        self.new_query_DE_STATS[self.mlca_iteration][economy_key][active_bidder]['init_pop_size'] = population_size

        # For first iteration of DE
        trial_population = population
        population_scw = [0] * population_size
        incumbent_allocation = None
        incumbent_scw = 0
        stop_counter = 0
        num_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
        logging.info(f'Running DE with {num_cpus} CPUs.')

        for i in range(1, iterations + 1):

            # DE Step 1: SELECTION
            # ----------------------------------------------------------------------------------------------------------------------------
            uUB_scw = self.calculate_uUB_scw(allocation=trial_population, economy_key=economy_key)
            exp_100_uUB_scw = self.calculate_exp_100_uUB_scw(allocation=trial_population, economy_key=economy_key)
            trial_population_scw = self.DE_parameters['weight_uUB_acquisition'] * uUB_scw + (
                    1 - self.DE_parameters['weight_uUB_acquisition']) * exp_100_uUB_scw
            # Select either the trial or the old population' individual depending on which is better
            # If the same take the new one for better exploration
            new_population = copy.deepcopy(population)
            new_population_scw = copy.deepcopy(population_scw)
            changed_individuals = 0
            logging.info('SELECTION START')
            for population_idx in range(population_size):
                # CHECK SCW:
                # --------------------------------------------------------------
                # l1 = np.sum(np.abs(convert_to_item_bidder_matrix(population_idx, population=population, num_items=self.M) - convert_to_item_bidder_matrix(population_idx, population=trial_population, num_items=self.M)))
                # print(f'Indiv: {population_idx} - L1: {l1:.3f} - SCW-Diff: {trial_population_scw[population_idx] - population_scw[population_idx]}')
                if trial_population_scw[population_idx] < population_scw[population_idx]:
                    self.new_query_DE_STATS[self.mlca_iteration][economy_key][active_bidder]['failed_CHECK_SCW'] += 1
                    continue
                # --------------------------------------------------------------

                # CHECK 1: IF ACTIVE BIDDER BUNDLE HAS BEEN ALREADY ELICITED
                # --------------------------------------------------------------
                if self.check_bundle_contained(bundle=trial_population[active_bidder][population_idx],
                                               bidder=active_bidder,
                                               log_level='DEBUG'):
                    self.new_query_DE_STATS[self.mlca_iteration][economy_key][active_bidder]['failed_CHECK1'] += 1
                    continue
                # --------------------------------------------------------------

                # CHECK 2: IF 100% SCW OF PROPOSED RANDOM ALLOCATION IS LARGER OR EQUAL TO CURRENT EFFICIENCY IN ECONOMY
                # --------------------------------------------------------------
                if not self.check_exp_100_uUB_SCW_is_larger(allocation=None,
                                                            economy_key=economy_key,
                                                            precalculated_exp_100_uUB_scw=exp_100_uUB_scw[
                                                            population_idx])[0]:
                    self.new_query_DE_STATS[self.mlca_iteration][economy_key][active_bidder]['failed_CHECK2'] += 1
                    continue
                # --------------------------------------------------------------
                if trial_population_scw[population_idx] > population_scw[population_idx]:
                    changed_individuals += 1
                for bidder in trial_population.keys():
                    new_population[bidder][population_idx] = trial_population[bidder][population_idx]
                    new_population_scw[population_idx] = trial_population_scw[population_idx]
            logging.info('SELECTION END')
            incumbent_scw, incumbent_idx = max(new_population_scw), np.argmax(new_population_scw)
            incumbent_allocation = {bidder: bundles[incumbent_idx] for bidder, bundles in new_population.items()}
            logging.info(
                'DE Iteration: {} | Current Incumbent SCW: {:.3f} | True Economy SCW: {:.3f} | Changed Indiv.: {}'.format(
                    i, incumbent_scw, self.get_current_economy_scw(economy_key), changed_individuals))
            self.new_query_DE_STATS[self.mlca_iteration][economy_key][active_bidder]['incumbent_trajectory'][i] = {
                'num_changed_individuals': changed_individuals,
                'incumbent_scw': incumbent_scw
            }

            population = new_population
            population_scw = new_population_scw

            # Early Stopping
            if changed_individuals == 0:
                stop_counter += 1
                if stop_counter >= patience_de_iterations:
                    logging.info('Early Stopping DE')
                    break
            elif i == iterations:
                # last iteration
                logging.info('Reached final iteration of DE')
                break
            else:
                stop_counter = 0  # reinitialize
            # ----------------------------------------------------------------------------------------------------------------------------

            # DE Step 2: MUTATION STRATEGY
            # ----------------------------------------------------------------------------------------------------------------------------
            convert_to_mat = partial(convert_to_item_bidder_matrix, population=population, num_items=self.M)
            if mutation_strategy == 'rand_1':
                def mutation_func(population_idx, population_size, incumbent_idx):
                    # Drop individual idx from individuals to consider for mutation
                    sampling_index = list(range(population_size))
                    sampling_index.remove(population_idx)

                    # Select three different individuals from the population
                    x_1_idx, x_2_idx, x_3_idx = random.sample(sampling_index, 3)

                    return convert_to_mat(x_1_idx) + random.random() * mutation_strength * (
                            convert_to_mat(x_2_idx) - convert_to_mat(x_3_idx))
            elif mutation_strategy == 'best_1':
                def mutation_func(population_idx, population_size, incumbent_idx):
                    # Drop individual idx from individuals to consider for mutation
                    sampling_index = list(range(population_size))
                    if population_idx != incumbent_idx:
                        sampling_index.remove(incumbent_idx)
                        sampling_index.remove(population_idx)
                    else:
                        sampling_index.remove(incumbent_idx)

                    # Select three different individuals from the population
                    x_2_idx, x_3_idx = random.sample(sampling_index, 2)

                    return convert_to_mat(incumbent_idx) + random.random() * mutation_strength * (
                            convert_to_mat(x_2_idx) - convert_to_mat(x_3_idx))
            elif mutation_strategy == 'rand_1_SCW_sampling':
                def mutation_func(population_idx, population_size, incumbent_idx):
                    # Drop individual idx from individuals to consider for mutation
                    sampling_index = list(range(population_size))
                    sampling_index.remove(population_idx)

                    # Select three different individuals from the population
                    scw_pos = np.asarray([population_scw[x] for x in sampling_index])
                    p_pos = scw_pos / sum(scw_pos)
                    x_1_idx, x_2_idx = np.random.choice(sampling_index,
                                                        size=2,
                                                        replace=False,
                                                        p=p_pos)
                    sampling_index.remove(x_1_idx)
                    sampling_index.remove(x_2_idx)
                    scw_neg = np.asarray([population_scw[x] for x in sampling_index])
                    p_neg = (1 / scw_neg) / sum(1 / scw_neg)
                    x_3_idx, = np.random.choice(sampling_index,
                                                size=1,
                                                replace=False,
                                                p=p_neg)
                    return convert_to_mat(x_1_idx) + random.random() * mutation_strength * (
                            convert_to_mat(x_2_idx) - convert_to_mat(x_3_idx))
            else:
                raise NotImplementedError('Unknown mutation strategy {}'.format(mutation_strategy))
            # ----------------------------------------------------------------------------------------------------------------------------
            logging.info('MUTATION - START')
            mutation_func_wrapped = \
                globalize(partial(mutation_func, population_size=population_size, incumbent_idx=incumbent_idx))
            pool = multiprocessing.Pool(num_cpus)
            mutants = pool.map(mutation_func_wrapped, list(range(population_size)))
            pool.close()  # terminate worker processes when all work already assigned has completed
            pool.join()  # wait all processes to terminate
            logging.info('MUTATION - END')

            # DE STEP 3: REPROJECTION
            # ----------------------------------------------------------------------------------------------------------------------------
            if reprojection_strategy == 'round-clip':
                raise NotImplementedError('Todo implement round-clip for reprojection')
            elif reprojection_strategy == 'bin':
                raise NotImplementedError('Todo implement binning for reprojection')
            elif reprojection_strategy == 'softmax-sampling':
                logging.info('REPROJECTION - START')
                # Each mutant is in [num_items, num_bidders] representation
                mutants = np.asarray(mutants)
                bidder_sampling_matrix = np.array(
                    torch.nn.functional.softmax(torch.from_numpy(mutants / softmax_temperature), dim=1))
                mutants = parallel_apply_along_axis(globalize(lambda pval: np.random.multinomial(1, pval)),
                                                    1, bidder_sampling_matrix, num_cpus=num_cpus)
                logging.info('REPROJECTION - LIST COMP.')
                mutants = [mutants[i] for i in range(len(mutants))]
                logging.info('REPROJECTION - END')

            else:
                raise NotImplementedError('Unknown reprojection strategy {}'.format(reprojection_strategy))
            # ----------------------------------------------------------------------------------------------------------------------------

            # DE Step 4: CROSS OVER STRATEGY
            # ----------------------------------------------------------------------------------------------------------------------------
            trial_population = copy.deepcopy(population)
            if crossover_strategy == 'bin':
                logging.info('CROSSOVER - START')
                for population_idx in range(population_size):
                    mutant = mutants[population_idx]
                    individual = convert_to_mat(population_idx)
                    crossover_points = np.random.rand(self.M) < crossover_prob
                    trial = copy.deepcopy(individual)
                    trial[:, crossover_points] = mutant[:, crossover_points]
                    for bidder_idx, bidder in enumerate(sorted(population.keys())):
                        trial_population[bidder][population_idx] = trial[bidder_idx, :]
                logging.info('CROSSOVER - END')
            elif crossover_strategy == 'exp':
                raise NotImplementedError('Todo implement exp cross over strategy.')
            else:
                raise NotImplementedError('Unknown crossover strategy {}'.format(crossover_strategy))
            # ----------------------------------------------------------------------------------------------------------------------------

        self.new_query_DE_STATS[self.mlca_iteration][economy_key][active_bidder][
            'true_scw'] = self.get_current_economy_scw(economy_key)
        self.new_query_DE_STATS[self.mlca_iteration][economy_key][active_bidder]['incumbent_eval_scw'] = incumbent_scw
        self.new_query_DE_STATS[self.mlca_iteration][economy_key][active_bidder]['iterations'] = i
        logging.info('STATISTICS:')
        for k, v in self.new_query_DE_STATS[self.mlca_iteration][economy_key][active_bidder].items():
            logging.info(f'->{k}:{v}')
        self.DE_allocation_cache_for_reusing[economy_key].append(incumbent_allocation)
        logging.debug(f'SAVING:{self.DE_allocation_cache_for_reusing[economy_key]}')
        self.number_of_optimization["bidder_specific_DE"] += 1

        return incumbent_allocation[active_bidder].reshape(-1, ).astype(
            int)  # reshape such that fits format of update_current_query_profile

    def get_random_allocation(self,
                              economy_key
                              ):

        '''
        Generates a single random allocation.
        NOTE: this is used in get_new_RS_bundle & get_new_DE_bundle
        '''

        random_idx = dict(list((bidder, []) for bidder in self.economies_names[economy_key]))
        available_items = list(range(0, self.M))
        while available_items:
            item = available_items.pop()
            bidder = np.random.choice(self.economies_names[economy_key], size=1)[0]
            random_idx[bidder].append(item)
        random_allocation = {}
        for bidder, items in random_idx.items():
            tmp = np.zeros(self.M)
            tmp[items] = 1
            random_allocation[bidder] = tmp.reshape(1, -1)

        return random_allocation

    def get_new_RS_bundle(self, economy_key, active_bidder, attempts=10000):

        logging.info('')
        logging.info(f'GENERATE NEW RS BUNDLE in {economy_key} for {active_bidder}:')
        logging.info('-----------------------------------------------')

        # Initialize new_random_exp_100_uUB_bundle save statistics
        if not self.new_query_RS_STATS.get(self.mlca_iteration):
            self.new_query_RS_STATS[self.mlca_iteration] = {}
        if not self.new_query_RS_STATS[self.mlca_iteration].get(economy_key):
            self.new_query_RS_STATS[self.mlca_iteration][economy_key] = {}
        if not self.new_query_RS_STATS[self.mlca_iteration][economy_key].get(active_bidder):
            self.new_query_RS_STATS[self.mlca_iteration][economy_key][active_bidder] = {
                'failed': False,
                'failed_CHECK1': 0,
                'failed_CHECK2': 0,
                'true_scw': None,
                'incumbent_eval_scw': None,
                'incumbent_exp_100_uUB_scw': None,
                'reused': False}

        # REUSE 1: Check if we can use bundle from random_allocation generated for a different (previous) bidder in this economy
        for prev_allocation in self.RS_allocation_cache_for_reusing[economy_key]:
            logging.debug(f'RS ALLOC FROM {economy_key}:{prev_allocation}')
            logging.debug(f'{active_bidder} BUNDLE:{prev_allocation[active_bidder].reshape(-1, )}')
            # CHECK 1 (only needed here since CHECK 2 has already been performed): IF ACTIVE BIDDER BUNDLE HAS BEEN ALREADY ELICITED
            # --------------------------------------------------------------
            if not self.check_bundle_contained(bundle=prev_allocation[active_bidder], bidder=active_bidder,
                                               log_level='DEBUG'):
                logging.info('REUSE BUNDLE FROM A PREVIOUS RS ALLOC IN THIS ECONOMY AND ITERATION')
                self.new_query_RS_STATS[self.mlca_iteration][economy_key][active_bidder]['reused'] = True
                logging.info('STATISTICS:')
                for k, v in self.new_query_RS_STATS[self.mlca_iteration][economy_key][
                    active_bidder].items():
                    logging.info(f'->{k}:{v}')
                self.number_of_optimization["bidder_specific_RS_REUSED"] += 1
                return prev_allocation[active_bidder].reshape(-1, ).astype(int)
            # --------------------------------------------------------------

        incumbent_allocation = None
        incumbent_scw = 0
        for i in range(1, attempts + 1):
            logging.debug(f'Attempt:{i}')

            random_allocation = self.get_random_allocation(economy_key=economy_key)

            # CHECK 1: IF ACTIVE BIDDER BUNDLE HAS BEEN ALREADY ELICITED
            # --------------------------------------------------------------
            if self.check_bundle_contained(bundle=random_allocation[active_bidder], bidder=active_bidder,
                                           log_level='DEBUG'):
                self.new_query_RS_STATS[self.mlca_iteration][economy_key][active_bidder]['failed_CHECK1'] += 1
                continue
            # --------------------------------------------------------------

            # CHECK 2: IF 100% SCW OF PROPOSED RANDOM ALLOCATION IS LARGER OR EQUAL TO CURRENT EFFICIENCY IN ECONOMY
            # --------------------------------------------------------------
            CHECK2, exp_100_uUB_scw, true_scw = self.check_exp_100_uUB_SCW_is_larger(allocation=random_allocation,
                                                                                     economy_key=economy_key,
                                                                                     precalculated_exp_100_uUB_scw=None)
            if not CHECK2:
                self.new_query_RS_STATS[self.mlca_iteration][economy_key][active_bidder]['failed_CHECK2'] += 1
                continue
            # --------------------------------------------------------------

            # UPDATE STEP:
            # --------------------------------------------------------------
            uUB_scw = self.calculate_uUB_scw(allocation=random_allocation, economy_key=economy_key)[0]
            acquisition_scw = self.RS_parameters['weight_uUB_acquisition'] * uUB_scw + (
                    1 - self.RS_parameters['weight_uUB_acquisition']) * exp_100_uUB_scw
            if acquisition_scw > incumbent_scw:
                incumbent_allocation = random_allocation
                incumbent_scw = acquisition_scw
                incumbent_exp_100_uUB_scw = exp_100_uUB_scw
            # --------------------------------------------------------------

        if incumbent_allocation:
            self.new_query_RS_STATS[self.mlca_iteration][economy_key][active_bidder]['true_scw'] = true_scw
            self.new_query_RS_STATS[self.mlca_iteration][economy_key][active_bidder][
                'incumbent_eval_scw'] = incumbent_scw
            self.new_query_RS_STATS[self.mlca_iteration][economy_key][active_bidder][
                'incumbent_exp_100_uUB_scw'] = incumbent_exp_100_uUB_scw
            logging.info('STATISTICS:')
            for k, v in self.new_query_RS_STATS[self.mlca_iteration][economy_key][
                active_bidder].items():
                logging.info(f'->{k}:{v}')
            self.RS_allocation_cache_for_reusing[economy_key].append(incumbent_allocation)
            logging.debug(f'SAVING:{self.RS_allocation_cache_for_reusing[economy_key]}')
            self.number_of_optimization['bidder_specific_RS'] += 1
            return incumbent_allocation[active_bidder].reshape(-1, ).astype(
                int)  # reshape such that fits format of update_current_query_profile

        logging.info(f'FAILED in {i} ATTEMPTS -> RETURN None')
        logging.info('STATISTICS:')
        for k, v in self.new_query_RS_STATS[self.mlca_iteration][economy_key][active_bidder].items():
            logging.info(f'->{k}:{v}')
        self.new_query_RS_STATS[self.mlca_iteration][economy_key][active_bidder]['failed'] = True
        return None

    def next_queries(self,
                     economy_key,
                     active_bidder):

        # CHECK IF ECONOMY HAS ALREADY BEEN CALCULATED IN THIS ITERATION FOR SOME DIFFERENT active_bidder
        if not self.economy_status[economy_key]:

            # STEP 1: Estimation of ML-models based on R (=elicited bids)
            # ----------------------------------------------------------------------------
            if self.separate_economy_training:  # only fit here if separate_economy_training=True
                self.estimation_step_economy(economy_key=economy_key)
            # ----------------------------------------------------------------------------

            # STEP 2: Optimization of trained ML-models
            # ----------------------------------------------------------------------------
            self.optimization_step(economy_key=economy_key,
                                   model_type='uUB_model',
                                   bidder_specific_constraints=None)
            # ----------------------------------------------------------------------------

        # NEW BUNDLE HAS BEEN ALREADY QUERIED IN R (=elicited bids from previous iterations) OR S (=elicited bids in current iteration)
        if self.check_bundle_contained(bundle=self.argmax_allocation[economy_key][active_bidder][0],
                                       bidder=active_bidder):

            if self.current_query_profile[active_bidder] is not None:
                Ri_union_Si = np.append(self.elicited_bids[active_bidder][0], self.current_query_profile[active_bidder],
                                        axis=0)
            else:
                Ri_union_Si = self.elicited_bids[active_bidder][0]

            # OPTION 1: BIDDER SPECIFIC MIP WITH uUB_model to generate NEW QUERY
            # ----------------------------------------------------------------------------
            if self.new_query_option == 'restricted_uUB_model_MIP':
                self.optimization_step(economy_key,
                                       model_type='uUB_model',
                                       bidder_specific_constraints={active_bidder: Ri_union_Si})
            # ----------------------------------------------------------------------------

            # OPTION 2: BIDDER SPECIFIC MIP WITH 100%-explicit UBs to generate new query
            # ----------------------------------------------------------------------------
            elif self.new_query_option == 'restricted_exp_100_uUB_model_MIP':
                self.optimization_step(economy_key,
                                       model_type='exp_100_uUB_model',
                                       bidder_specific_constraints={active_bidder: Ri_union_Si})
            # ----------------------------------------------------------------------------

            # OPTION 3: BIDDER SPECIFIC RS BUNDLE
            # ----------------------------------------------------------------------------
            elif self.new_query_option == 'RS':
                q_new = self.get_new_RS_bundle(economy_key=economy_key,
                                               active_bidder=active_bidder,
                                               attempts=self.RS_parameters['attempts'])
                if q_new is None:
                    self.optimization_step(economy_key,
                                           model_type='uUB_model',
                                           bidder_specific_constraints={active_bidder: Ri_union_Si})
                else:
                    self.argmax_allocation[economy_key][active_bidder][1] = q_new
            # ----------------------------------------------------------------------------

            # OPTION 4: BIDDER SPECIFIC DE BUNDLE
            # ----------------------------------------------------------------------------
            elif self.new_query_option == 'DE':
                q_new = self.get_new_DE_bundle(economy_key=economy_key,
                                               active_bidder=active_bidder,
                                               iterations=self.DE_parameters['iterations'])
                if q_new is None:
                    self.optimization_step(economy_key,
                                           model_type='uUB_model',
                                           bidder_specific_constraints={active_bidder: Ri_union_Si})
                else:
                    self.argmax_allocation[economy_key][active_bidder][1] = q_new
            # ----------------------------------------------------------------------------

            self.economy_status[economy_key] = True  # set status of economy to true
            self.print_argmax_allocation(economy_key)
            return self.argmax_allocation[economy_key][active_bidder][1]  # return constrained argmax bundle


        # NEW BUNDLE HAS NOT BEEN ALREADY QUERIED: accept it
        else:
            logging.info('ARGMAX BUNDLE FROM NORMAL MIP CAN BE USED')
            self.economy_status[economy_key] = True  # set status of economy to true
            self.print_argmax_allocation(economy_key)

            return self.argmax_allocation[economy_key][active_bidder][0]  # return regular argmax bundle

    def estimation_step_economy(self,
                                economy_key):

        start_estimation = datetime.now()
        logging.info(f'ESTIMATION STEP:{economy_key}')
        logging.info('-----------------------------------------------')

        training_seeds = self.update_nn_training_seeds(number_of_seeds=len(self.economies_names[economy_key]))

        # OPTION 1: PARALLEL TRAINING
        # ----------------------------------------------------------------------------
        if self.parallelize_training:
            pool_args = zip(self.economies_names[economy_key], training_seeds)
            m = Parallel(n_jobs=-1)(delayed(
                partial(eval_bidder_nn,
                        fitted_scaler=self.fitted_scaler,
                        NN_parameters=self.NN_parameters,
                        elicited_bids=self.elicited_bids,
                        local_scaling_factor=self.local_scaling_factor,
                        num_cpu_per_job=1
                        )
            )(bidder, seed) for bidder, seed in pool_args)
            models = merge_dicts(m)
        # ----------------------------------------------------------------------------

        # OPTION 2: SEQUENTIAL TRAINING
        # ----------------------------------------------------------------------------
        else:
            models = []
            for bidder, seed in zip(self.economies_names[economy_key], training_seeds):
                models.append(eval_bidder_nn(fitted_scaler=self.fitted_scaler,
                                             NN_parameters=self.NN_parameters,
                                             elicited_bids=self.elicited_bids,
                                             local_scaling_factor=self.local_scaling_factor,
                                             bidder=bidder,
                                             seed=seed, num_cpu_per_job=os.cpu_count())
                              )
            models = merge_dicts(models)
        # ----------------------------------------------------------------------------

        # Add train metrics
        trained_models = {}
        if not self.train_logs.get(self.mlca_iteration):
            self.train_logs[self.mlca_iteration] = OrderedDict(list(
                (key, OrderedDict(list((bidder_id, []) for bidder_id in value))) for key, value in
                self.economies_names.items()))

        for bidder_id, (model, train_logs) in models.items():
            trained_models[bidder_id] = model
            self.train_logs[self.mlca_iteration][economy_key][bidder_id] = train_logs
            # total time for ml training
            self.ml_total_train_time_sec += train_logs["train_time_elapsed"]

        self.NN_models[economy_key] = trained_models

        end_estimation = datetime.now()
        self.total_time_elapsed_estimation_step += (end_estimation - start_estimation).total_seconds()
        logging.info('Elapsed Time: {}d {}h:{}m:{}s\n'.format(*timediff_d_h_m_s(end_estimation - start_estimation)))
        return

    def estimation_step(self):

        start_estimation = datetime.now()
        logging.info('ESTIMATION STEP: all economies')
        logging.info('-----------------------------------------------')

        training_seeds = self.update_nn_training_seeds(number_of_seeds=len(self.bidder_names))

        # OPTION 1: PARALLEL TRAINING
        # ----------------------------------------------------------------------------
        if self.parallelize_training:
            pool_args = zip(self.bidder_names, training_seeds)
            m = Parallel(n_jobs=-1)(delayed(partial(eval_bidder_nn,
                                                    fitted_scaler=self.fitted_scaler,
                                                    NN_parameters=self.NN_parameters,
                                                    elicited_bids=self.elicited_bids,
                                                    local_scaling_factor=self.local_scaling_factor,
                                                    num_cpu_per_job=1
                                                    )
                                            )(bidder, seed) for bidder, seed in pool_args
                                    )
            models = merge_dicts(m)
            # ----------------------------------------------------------------------------

            # OPTION 2: SEQUENTIAL TRAINING
            # ----------------------------------------------------------------------------
        else:
            models = []
            for bidder, seed in zip(self.bidder_names, training_seeds):
                models.append(eval_bidder_nn(fitted_scaler=self.fitted_scaler, NN_parameters=self.NN_parameters,
                                             elicited_bids=self.elicited_bids,
                                             local_scaling_factor=self.local_scaling_factor,
                                             bidder=bidder,
                                             seed=seed, num_cpu_per_job=os.cpu_count()))
            models = merge_dicts(models)
            # ----------------------------------------------------------------------------

        # Add train metrics
        trained_models = {}
        if not self.train_logs.get(self.mlca_iteration):
            self.train_logs[self.mlca_iteration] = OrderedDict(
                list(('Bidder_{}'.format(bidder_id), []) for bidder_id in self.bidder_ids))

        for bidder_id, (model, train_logs) in models.items():
            trained_models[bidder_id] = model
            self.train_logs[self.mlca_iteration][bidder_id] = train_logs
            # total time for ml training
            self.ml_total_train_time_sec += train_logs["train_time_elapsed"]

        self.NN_models = trained_models

        end_estimation = datetime.now()
        self.total_time_elapsed_estimation_step += (end_estimation - start_estimation).total_seconds()
        logging.info('Elapsed Time: {}d {}h:{}m:{}s\n'.format(*timediff_d_h_m_s(end_estimation - start_estimation)))
        return

    def update_nn_training_seeds(self,
                                 number_of_seeds
                                 ):

        training_seeds = list(range(self.nn_seed, self.nn_seed + number_of_seeds))
        self.nn_seed += number_of_seeds  # update

        return training_seeds

    def get_ML_models(self,
                      economy_key,
                      model_type):

        '''
        # Extract pytorch ML models from self.NN_models[...][...] via:
        # --------------------------------------------
        # 1. Upper-UB:          .uUB_model
        # 2. Mean:              .mean_model
        # 3. 100%-explicit-uUB: .exp_100_uUB_model
        # --------------------------------------------
        '''

        MODELS = OrderedDict()

        if model_type == 'uUB_model':

            for bidder in self.economies_names[economy_key]:

                if self.separate_economy_training:
                    assert self.economies_names[economy_key] == list(self.NN_models[economy_key].keys())
                    MODELS[bidder] = self.NN_models[economy_key][bidder].uUB_model
                else:
                    MODELS[bidder] = self.NN_models[bidder].uUB_model

        elif model_type == 'mean_model':

            for bidder in self.economies_names[economy_key]:

                if self.separate_economy_training:
                    assert self.economies_names[economy_key] == list(self.NN_models[economy_key].keys())
                    MODELS[bidder] = self.NN_models[economy_key][bidder].mean_model
                else:
                    MODELS[bidder] = self.NN_models[bidder].mean_model

        elif model_type == 'exp_100_uUB_model':

            for bidder in self.economies_names[economy_key]:

                if self.separate_economy_training:
                    assert self.economies_names[economy_key] == list(self.NN_models[economy_key].keys())
                    MODELS[bidder] = self.NN_models[economy_key][bidder].exp_100_uUB_model
                else:
                    MODELS[bidder] = self.NN_models[bidder].exp_100_uUB_model

        else:
            raise NotImplementedError(
                f'model_type:{model_type} not implemented. Select from uUB_model, mean_model, or exp_100_uUB_model.')

        return MODELS

    def optimization_step(self,
                          economy_key,
                          model_type,
                          bidder_specific_constraints=None
                          ):

        # Extract pytorch models
        MODELS = self.get_ML_models(economy_key, model_type=model_type)

        if bidder_specific_constraints is None:
            logging.info('OPTIMIZATION STEP')
        else:
            logging.info(
                f'ADDITIONAL BIDDER SPECIFIC **{self.new_query_option}** for {list(bidder_specific_constraints.keys())[0]}')
        logging.info('-----------------------------------------------')

        attempts = self.MIP_parameters['attempts_DNN_WDP']

        # NEW GSVM specific constraints
        if (self.SATS_auction_instance.get_model_name() == 'GSVM' and not self.SATS_auction_instance.isLegacy):
            GSVM_specific_constraints = True
            national_circle_complement = list(self.good_ids - set(self.SATS_auction_instance.get_goods_of_interest(6)))
            logging.info('########## ATTENTION ##########')
            logging.info('GSVM specific constraints: %s', GSVM_specific_constraints)
            logging.info('###############################\n')
        else:
            GSVM_specific_constraints = False
            national_circle_complement = None

        for attempt in range(1, attempts + 1):

            # counter
            if bidder_specific_constraints:
                self.number_of_optimization['bidder_specific_MIP'] += 1
            else:
                self.number_of_optimization['normal'] += 1

            logging.debug('Initialize MIP')

            # NEW MVNN-MIP
            MIP = MVNN_MIP_TORCH_NEW(MODELS)
            MIP.initialize_mip(verbose=False,
                               bidder_specific_constraints=bidder_specific_constraints,
                               GSVM_specific_constraints=GSVM_specific_constraints,  # NEW
                               national_circle_complement=national_circle_complement)  # NEW

            try:
                logging.info('Solving MIP')
                logging.info('Attempt no: %s', attempt)
                if self.MIP_parameters['warm_start'] and self.warm_start_sol[economy_key] is not None:
                    logging.debug('Using warm start')
                    sol, log = MIP.solve_mip(
                        log_output=False,
                        time_limit=self.MIP_parameters['time_limit'],
                        mip_relative_gap=self.MIP_parameters['relative_gap'],
                        integrality_tol=self.MIP_parameters['integrality_tol'],
                        feasibility_tol=self.MIP_parameters['feasibility_tol'],
                        mip_start=docplex.mp.solution.SolveSolution(MIP.Mip,
                                                                    self.warm_start_sol[economy_key].as_dict()))
                    self.warm_start_sol[economy_key] = sol
                else:
                    sol, log = MIP.solve_mip(
                        log_output=False,
                        time_limit=self.MIP_parameters['time_limit'],
                        mip_relative_gap=self.MIP_parameters['relative_gap'],
                        integrality_tol=self.MIP_parameters['integrality_tol'],
                        feasibility_tol=self.MIP_parameters['feasibility_tol'])
                    self.warm_start_sol[economy_key] = sol

                for key, value in log.items():
                    self.mip_logs[key].append(value)

                if bidder_specific_constraints is None:
                    logging.debug('SET ARGMAX ALLOCATION FOR ALL BIDDERS')
                    b = 0
                    for bidder in self.argmax_allocation[economy_key].keys():
                        self.argmax_allocation[economy_key][bidder][0] = MIP.x_star[b, :]
                        b = b + 1
                else:
                    logging.debug(
                        f'SET ARGMAX ALLOCATION ONLY BIDDER SPECIFIC for {list(bidder_specific_constraints.keys())[0]}')
                    for bidder in bidder_specific_constraints.keys():
                        b = MIP.get_bidder_key_position(
                            bidder_key=bidder)  # transform bidder_key into bidder position in MIP
                        self.argmax_allocation[economy_key][bidder][1] = MIP.x_star[b, :]  # now on position 1!

                self.elapsed_time_mip[economy_key].append(MIP.soltime)
                break

            except Exception:
                logging.warning('-----------------------------------------------')
                logging.warning('NOT SUCCESSFULLY SOLVED in attempt: %s \n', attempt)
                logging.warning(MIP.Mip.solve_details)
                if attempt == attempts:
                    MIP.Mip.export_as_lp(basename='UnsolvedMip_iter{}_{}'.format(self.mlca_iteration, economy_key),
                                         path=os.getcwd(), hide_user_names=False)
                    sys.exit('STOP, not solved succesfully in {} attempts\n'.format(attempt))

                logging.debug('REFITTING:')

                # self.separate_economy_training=True -> only refit for this economy
                if self.separate_economy_training:
                    self.estimation_step_economy(economy_key=economy_key)

                # self.separate_economy_training=False -> Re-fit all irrespective of economies
                else:
                    self.estimation_step()

                MODELS = self.get_ML_models(economy_key, model_type=model_type)

        del MIP
        del MODELS

    def calculate_mlca_allocation(self,
                                  economy='Main Economy'
                                  ):

        '''
        REMARK: objective always rescaled to true original values
        '''

        logging.info('Calculate MLCA allocation: %s', economy)
        active_bidders = self.economies_names[economy]
        logging.debug('Active bidders: %s', active_bidders)
        allocation, objective = self.solve_WDP(
            elicited_bids=OrderedDict(list((k, self.elicited_bids.get(k, None)) for k in active_bidders)))
        logging.debug('MLCA allocation in %s:', economy)
        for key, value in allocation.items():
            logging.debug('%s %s', key, value)
        logging.debug('Social Welfare: %s', objective)
        # setting allocations
        if economy == 'Main Economy':
            self.mlca_allocation = allocation
            self.mlca_scw = objective
        if economy in self.mlca_marginal_allocations.keys():
            self.mlca_marginal_allocations[economy] = allocation
            self.mlca_marginal_scws[economy] = objective
            self.mlca_efficiency_marginals[economy] = objective / self.SATS_auction_instance_scw

    def solve_WDP(self,
                  elicited_bids,
                  verbose=0
                  ):

        '''
        REMARK: objective always rescaled to true original values
        '''

        bidder_names = list(elicited_bids.keys())
        if verbose == 1: logging.debug('Solving WDP based on elicited bids for bidder: %s', bidder_names)
        elicited_bundle_value_pairs = [np.concatenate((bids[0], np.asarray(bids[1]).reshape(-1, 1)), axis=1) for
                                       bidder, bids in
                                       elicited_bids.items()]  # transform self.elicited_bids into format for WDP class
        wdp = MLCA_WDP(elicited_bundle_value_pairs)
        wdp.initialize_mip(verbose=0)
        wdp.solve_mip(verbose)

        objective = wdp.Mip.objective_value
        allocation = format_solution_mip_new(Mip=wdp.Mip, elicited_bids=elicited_bundle_value_pairs,
                                             bidder_names=bidder_names, fitted_scaler=self.fitted_scaler)
        if self.fitted_scaler is not None:
            if verbose == 1:
                logging.debug('')
                logging.debug('*SCALING*')
                logging.debug('---------------------------------------------')
                logging.debug('WDP objective scaled: %s:', objective)
                logging.debug('WDP objective value scaled by: 1/%s', round(self.fitted_scaler.scale_[0], 8))
            objective = float(self.fitted_scaler.inverse_transform([[objective]]))
            if verbose == 1:
                logging.debug('WDP objective orig: %s:', objective)
                logging.debug('---------------------------------------------')
        return (allocation, objective)

    def calculate_efficiency_of_allocation(self,
                                           allocation,
                                           allocation_scw,
                                           verbose=0
                                           ):

        self.solve_SATS_auction_instance()
        efficiency = allocation_scw / self.SATS_auction_instance_scw
        if verbose == 1:
            logging.debug('Calculating efficiency of input allocation:')
            for key, value in allocation.items():
                logging.debug('%s %s', key, value)
            logging.debug('Social Welfare: %s', allocation_scw)
            logging.debug('Efficiency of allocation: %s', efficiency)
        return (efficiency)

    def calculate_vcg_payments(self,
                               forced_recalc=False
                               ):

        logging.debug('Calculate payments')
        # (i) solve marginal MIPs
        for economy in list(self.economies_names.keys()):
            if not forced_recalc:
                if economy == 'Main Economy' and self.mlca_allocation is None:
                    self.calculate_mlca_allocation()
                elif economy in self.mlca_marginal_allocations.keys() and self.mlca_marginal_allocations[
                    economy] is None:
                    self.calculate_mlca_allocation(economy=economy)
                else:
                    logging.debug('Allocation for %s already calculated', economy)
            else:
                logging.debug('Forced recalculation of %s', economy)
                self.calculate_mlca_allocation(economy=economy)  # Recalc economy

        # (ii) calculate VCG terms for this economy
        for bidder in self.bidder_names:
            marginal_economy_bidder = 'Marginal Economy -({})'.format(key_to_int(bidder))
            p1 = self.mlca_marginal_scws[
                marginal_economy_bidder]  # social welfare of the allocation a^(-i) in this economy
            p2 = sum([self.mlca_allocation[i]['value'] for i in self.economies_names[
                marginal_economy_bidder]])  # social welfare of mlca allocation without bidder i
            self.mlca_payments[bidder] = round(p1 - p2, 2)
            logging.info('Payment %s: %s - %s  =  %s', bidder, p1, p2, self.mlca_payments[bidder])
        self.revenue = sum([self.mlca_payments[i] for i in self.bidder_names])
        self.relative_revenue = self.revenue / self.SATS_auction_instance_scw
        logging.info(
            'Revenue: {} | {}% of SCW in efficient allocation\n'.format(self.revenue, self.relative_revenue * 100))
