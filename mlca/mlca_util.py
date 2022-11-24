"""
FILE DESCRIPTION:

This file stores helper functions used across the files in this project.

"""

import copy
import logging
import math
import random
import re
from collections import OrderedDict, namedtuple

# Libs
import numpy as np
from scipy.stats import binom

from mlca.mlca_value_model import ValueModel

instance_info = namedtuple('instance_info', ['bidder_types', 'Qmax', 'Qround', 'Qinit'])

problem_instance_info = {
    'GSVM': instance_info(bidder_types=['Regional', 'National'], Qmax=100, Qround=4, Qinit=40),
    'LSVM': instance_info(bidder_types=['Regional', 'National'], Qmax=100, Qround=4, Qinit=40),
    'MRVM': instance_info(bidder_types=['Local', 'Regional', 'National'], Qmax=100, Qround=4, Qinit=40),
    'SRVM': instance_info(bidder_types=['Local', 'Regional', 'National', 'High_Frequency'], Qmax=100, Qround=4, Qinit=40)
}
# %% (0) HELPER FUNCTIONS
# %% replicate configs with seeds_instance

def helper_f(CONFIGS):
    tmp1 = []
    for c in CONFIGS:
        tmp2 = []
        for seed in c['SATS_auction_instance_seeds']:
            y = copy.deepcopy(c)
            y['SATS_auction_instance_seed'] = seed
            tmp2.append(y)
            del y
        tmp1.append(tmp2)
    return (tmp1)


# %%
def pretty_print_dict(D, printing=True):
    text = []
    for key, value in D.items():
        if key in ['NN_parameters', 'MIP_parameters']:
            if printing:
                print(key, ': ')
            text.append(key + ': \n')
            for k, v in value.items():
                if printing:
                    print(k + ': ', v)
                text.append(k + ': ' + str(v) + '\n')
        else:
            if printing:
                print(key, ':  ', value)
            text.append(key + ':  ' + str(value) + '\n')
    return (''.join(text))


# %%
def timediff_d_h_m_s(td):
    # can also handle negative datediffs
    if (td).days < 0:
        td = -td
        return -(td.days), -int(td.seconds / 3600), -int(td.seconds / 60) % 60, -(td.seconds % 60)
    return td.days, int(td.seconds / 3600), int(td.seconds / 60) % 60, td.seconds % 60


# %% Tranforms bidder_key to integer bidder_id
# key = valid bidder_key (string), e.g. 'Bidder_0'
def key_to_int(key):
    return (int(re.findall(r'\d+', key)[0]))


# %% Pivotal Bootstrap Confidence Intervall [2\hat(theta)-Q_1-alpha/2, 2\hat(theta)+Q_alpha/2], see https://en.wikipedia.org/wiki/Bootstrapping_(statistics)


def boot_conf(data, alpha, number_of_bootstraps, f_statistic):
    theta_hat = f_statistic(data)
    statistics = []
    for i in range(number_of_bootstraps):
        sample = random.choices(data, k=len(data))
        stat = f_statistic(sample)
        statistics.append(stat)
    ordered = np.sort(statistics)
    lower = np.percentile(ordered, 100 * (alpha / 2))
    upper = np.percentile(ordered, 100 * (1 - alpha / 2))
    confidence_interval = (2 * theta_hat - upper, 2 * theta_hat - lower)
    return ({'stat': theta_hat, 'lower': lower, 'upper': upper, 'confidence Interval': confidence_interval})


# %% PREPARE INITIAL BIDS FOR A SINGLE INSTANCE FOR ALL BIDDERS for MLCA MECHANISM
# THIS METHOD USES SATS SAMPLING FROM ADMISSIBLE BUNDLE SPACE

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# SATS_auction_instance = single instance of a value model
# number_initial_bids = number of initial bids
# bidder_ids = bidder ids in this value model (int)
# scaler = scale the y values across all bidders, fit on the selected training set and apply on the validation set
# seed = seed for random initial bids

def initial_bids_mlca_unif(SATS_auction_instance,
                           number_initial_bids,
                           bidder_names,
                           scaler=None,
                           seed=None,
                           include_full_bundle=False):

    initial_bids = OrderedDict()

    if include_full_bundle:
        number_initial_bids -=1

    # seed determines bidder_seeds for all bidders, e.g. seed=10 and 3 bidders generates bidder_seeds=[28,29,30]
    n_bidders = len(bidder_names)
    if seed is not None:
        bidder_seeds = list(range((seed-1) * n_bidders+1, (seed) * n_bidders+1))
    else:
        bidder_seeds = [None] * n_bidders
    logging.debug(f'Bidder specific seeds for initial bundle-value pairs:{bidder_seeds}')

    i = 0
    for bidder in bidder_names:

        bidder_id = key_to_int(bidder)

        # Sampling method from SATS, which incorporates bidder specific restrictions:
        # e.g. in GSVM for regional bidders only bundles of up to size 4 are sampled and for national bidders only bundles that
        # contain items from the national-circle are sampled.
        # Remark: SATS does not ensure that bundles are unique, this needs to be taken care exogenously.
        # D = (X,y) in ({0,1}^m x R_+)^number_of_bids*(m+1)
        D = np.asarray(SATS_auction_instance.get_uniform_random_bids(bidder_id=bidder_id,
                                                                     number_of_bids=number_initial_bids,
                                                                     seed=seed),dtype=np.float32)

        # only use X from SATS generator, since then uniqueness check is easier
        X = D[:, :-1]

        M = len(SATS_auction_instance.get_good_ids())

        full_bundle = np.array([1]*M, dtype=np.float32)
        empty_bundle = np.array([0]*M, dtype=np.float32)

        # Remove full bundle and null bundle if they were drawn
        full_idx = np.where(np.all(X==full_bundle,axis=1))[0]
        empty_idx =  np.where(np.all(X==empty_bundle,axis=1))[0]
        if len(full_idx)>0:
            X = np.delete(X,full_idx,axis=0)
        if len(empty_idx)>0:
            X = np.delete(X,empty_idx,axis=0)
        #

        # UNIQUENESS
        X = np.unique(X,axis=0)
        seed_additional_bundle = None if seed is None else (10 ** 5) * seed
        while X.shape[0] != (number_initial_bids):
            logging.debug(f'Generate new bundle: only {X.shape[0]+1} are unique but you asked for:{number_initial_bids}')
            dnew = np.asarray(SATS_auction_instance.get_uniform_random_bids(bidder_id=bidder_id,number_of_bids=1,seed=seed_additional_bundle))
            xnew = dnew[:, :-1]
            # Check until new bundle is different from FULL_BUNDLE and NULL_BUNDLE
            while np.all(xnew==full_bundle) or np.all(xnew==empty_bundle):
                if seed_additional_bundle is not None: seed_additional_bundle += 1
                dnew = np.asarray(SATS_auction_instance.get_uniform_random_bids(bidder_id=bidder_id,number_of_bids=1,seed=seed_additional_bundle),dtype=np.float32)
                xnew = dnew[:, :-1]
                logging.debug(f'RESAMPLE additional bundle SINCE it equals null-bundle OR full-bundle:{xnew}')
            X = np.concatenate((X,xnew),axis=0)
            X = np.unique(X,axis=0)
            if seed_additional_bundle is not None: seed_additional_bundle += 1
        # --------------------------------------------

        # generate bidders' values for initial bundles
        y = np.array(SATS_auction_instance.calculate_values(bidder_id, X), dtype=np.float32)
        # --------------------------------------------

        # a) potentially include full bundle
        # --------------------------------------------
        if include_full_bundle:
            value_full_bundle = np.array([SATS_auction_instance.calculate_value(bidder_id, full_bundle)], dtype=np.float32)
            X = np.concatenate((X,full_bundle.reshape(1,-1)))
            y = np.concatenate((y,value_full_bundle))
        # --------------------------------------------

        # b) always include empty bundle since no value query needed, i.e., we know that it has value 0
        # --------------------------------------------
        value_empty_bundle = np.array([0.0], dtype=np.float32)
        X = np.concatenate((X,empty_bundle.reshape(1,-1)))
        y = np.concatenate((y,value_empty_bundle))
       # --------------------------------------------

        X = X.astype(int) # needed for MIP
        y = y.astype(np.float32)
        X, y = unison_shuffled_copies(X, y)

        assert len(np.unique(X,axis=0)) == len(X)
        logging.info('INIT BUNDLE-VALUE PAIRS ARE UNIQUE')
        logging.info(f'X in {X.shape}, y in {y.shape} (incl. empty-bundle) with include_full_bundle:{include_full_bundle}.')
        initial_bids[bidder] = [X, y]
        i += 1

    if scaler is not None:
        tmp = np.array([])
        for bidder in bidder_names:
            tmp = np.concatenate((tmp, initial_bids[bidder][1]), axis=0)
        scaler.fit(tmp.reshape(-1, 1))
        logging.debug('')
        logging.debug('*SCALING*')
        logging.debug('---------------------------------------------')
        logging.debug('Samples seen: %s', scaler.n_samples_seen_)
        logging.debug('Data max: %s', scaler.data_max_)
        logging.debug('Data min: %s', scaler.data_min_)
        logging.debug('Scaling by: %s | %s==feature range max?', scaler.scale_, float(scaler.data_max_ * scaler.scale_))
        logging.debug('---------------------------------------------')
        initial_bids = OrderedDict(list(
            (key, [value[0], scaler.transform(value[1].reshape(-1, 1)).flatten()]) for key, value in
            initial_bids.items()))

    return (initial_bids, scaler)

# %% This function formates the solution of the winner determination problem (WDP) given elicited bids.
# Mip = A solved DOcplex instance.
# elicited_bids = the set of elicited bids for each bidder corresponding to the WDP.
# bidder_names = bidder names (string, e.g., 'Bidder_1')
# fitted_scaler = the fitted scaler used in the valuation model.


def format_solution_mip_new(Mip, elicited_bids, bidder_names, fitted_scaler):
    tmp = {'good_ids': [], 'value': 0}
    Z = OrderedDict()
    for bidder_name in bidder_names:
        Z[bidder_name] = tmp
    S = Mip.solution.as_dict()
    for key in list(S.keys()):
        key = str(key)
        index = [int(x) for x in re.findall(r'\d+', key)]
        bundle = elicited_bids[index[0]][index[1], :-1]
        value = elicited_bids[index[0]][index[1], -1]
        if fitted_scaler is not None:
            logging.debug('*SCALING*')
            logging.debug('---------------------------------------------')
            logging.debug(value)
            logging.debug('WDP values for allocation scaled by: 1/%s', round(fitted_scaler.scale_[0], 8))
            value = float(fitted_scaler.inverse_transform([[value]]))
            logging.debug(value)
            logging.debug('---------------------------------------------')
        bidder = bidder_names[index[0]]
        Z[bidder] = {'good_ids': list(np.where(bundle == 1)[0]), 'value': value}
    return (Z)
# %%

def sample_item_of_size_k(k, m):
    tmp = [1 if idx < k else 0 for idx in range(m)]
    random.shuffle(tmp)
    return tmp


def _unif_pseudo_sampling(m, n):
    D = []
    q_init = n
    x = binom.pmf(list(range(m + 1)), m, 1 / 2) * (q_init + 1)  # check that this line is correct
    b = 0
    s = x[0] - 1

    while b < m:
        b += 1
        a = b
        s += x[b]

        while s < 1 and b < m:
            b += 1
            s += x[b]
        for j in range(0, math.floor(max(s, 1))):
            k = random.choices(list(range(a, b + 1)), weights=x[a:b + 1], k=1)[0]  # maybe weigh with x
            sample = sample_item_of_size_k(k, m)
            idx = 0
            while sample in D:  # Ensure that the initial samples are unique
                sample = sample_item_of_size_k(k, m)
                idx += 1
                assert idx < 1000, 'The sampling of unique bundles was unsuccessful.'
            D.append(sample)

        s -= math.floor(s)

    if sum(D[-1]) != m:
        D[-1] = [1] * m
    return D


def unif_pseudo_random_bids(value_model, seed, n, bidder_id):
    logging.debug('Sampling uniformly at random %s bundle-value pairs from bidder %s', n, bidder_id)

    if seed:
        random.seed(seed)

    D = _unif_pseudo_sampling(m=len(value_model.get_good_ids()), n=n)

    # define helper function for specific bidder_id
    def myfunc(bundle):
        return value_model.calculate_value(bidder_id, bundle)

    D = np.hstack((D, np.apply_along_axis(myfunc, 1, D).reshape(-1, 1)))
    del myfunc
    return (D)

# %%
def create_value_model(value_model):
    if value_model == 'LSVM':
        V = ValueModel(name='LSVM', number_of_items=18, local_bidder_ids=[], regional_bidder_ids=list(range(1, 6)),
                       national_bidder_ids=[0], scaler=[None], highFrequency_bidder_ids=[])
    elif value_model == 'GSVM':
        V = ValueModel(name='GSVM', number_of_items=18, local_bidder_ids=[], regional_bidder_ids=list(range(0, 6)),
                       national_bidder_ids=[6], scaler=[None], highFrequency_bidder_ids=[])
    elif value_model == 'MRVM':
        V = ValueModel(name='MRVM', number_of_items=98, local_bidder_ids=[0, 1, 2], regional_bidder_ids=[3, 4, 5, 6],
                       national_bidder_ids=[7, 8, 9], scaler=[None], highFrequency_bidder_ids=[])
    elif value_model == 'SRVM':
        V = ValueModel(name='SRVM', number_of_items=29, local_bidder_ids=[0, 1], highFrequency_bidder_ids=[2],
                       regional_bidder_ids=[3, 4], national_bidder_ids=[5, 6], scaler=[None])
    else:
        raise NotImplementedError('Unknown value model: {}'.format(value_model))
    return V
