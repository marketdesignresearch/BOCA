import argparse
import glob
import json
import logging
import os
import re
from collections import defaultdict
import sys
from sklearn.preprocessing import MinMaxScaler

from mlca.mlca_mechanism import mlca_mechanism
from mlca.mlca_util import create_value_model, problem_instance_info
from util import StreamToLogger


def load_HPO_winners(dir: str, domain: str, q: float):
    results = defaultdict(lambda: defaultdict(dict))
    for file in glob.glob(os.path.join(dir, '*')):
        if file.endswith('.json'):
            hpo_winner = json.load(open(file))
            config_key = [key for key in hpo_winner.keys() if 'config' in key][0]
            config = hpo_winner[config_key]
            q_eval = float(re.findall('0\.[0-9.]{1,2}', file)[0])
            config['layer_type'] = 'MVNNLayerReLUProjected'
            results[config['SATS_domain']][q_eval][config['bidder_type']] = config
    return results[domain][q]


def main(domain: str,
         q: float,
         seed: int
         ):


    res_path = os.path.join(os.getcwd(),
                            'results',
                             domain,
                             str(q),
                             str(seed)
                             )
    os.makedirs(res_path, exist_ok=False)

    

    # Clear existing logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # --------------------------------------

    # Define logger and log current STATUS_MSG
    logging.basicConfig(level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s: %(message)s',
                        filename=os.path.join(res_path, 'log.txt'),
                        filemode='w'
                        )
    log = logging.getLogger('log')
    # Nice solution to add the stdout and stderr to logging from here
    # https://stackoverflow.com/a/39215961
    sys.stdout = StreamToLogger(log, logging.INFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)

    # Loading HPO MVNN Winners
    NN_parameters = load_HPO_winners(os.path.join(os.getcwd(),'hpo_results_mvnnUB4','winner'),
                                     domain=domain,
                                     q=q)

    value_model = create_value_model(domain)
    NN_parameters = value_model.parameters_to_bidder_id(NN_parameters)
    scaler = MinMaxScaler(feature_range=(0, 500))
    Qmax, Qround, Qinit = problem_instance_info[value_model.name.upper()].Qmax, \
                          problem_instance_info[value_model.name.upper()].Qround, \
                          problem_instance_info[value_model.name.upper()].Qinit


    # NEW MVNN-uUB-MLCA PARAMETERS:
    # -------------------
    separate_economy_training = False  # mvnns trained seperately for each economy (default=True)
    new_query_option = 'restricted_uUB_model_MIP'  # from 'restricted_uUB_model_MIP','restricted_exp_100_uUB_model_MIP','RS'=random search
    balanced_global_marginals = True
    parallelize_training = True
    # -------------------

    # MIP-parameters
    MIP_parameters = {
        'bigM': 2000,
        'mip_bounds_tightening': None,
        'warm_start': False,
        'time_limit': 600,
        'relative_gap': 5e-3,
        'integrality_tol': 1e-6,
        'feasibility_tol': 1e-9,
        'attempts_DNN_WDP': 5
    }

    # RS-parameters
    RS_parameters = {
        'attempts': 10000,
        'weight_uUB_acquisition': 0.95
        }

    kwargs = {
        'SATS_domain_name': value_model.name.upper(),
        'SATS_auction_instance_seed': seed,
        'Qinit': Qinit,
        'Qmax': Qmax,
        'Qround': Qround,
        'separate_economy_training': separate_economy_training,  # NEW parameter
        'new_query_option': new_query_option,  # NEW parameter
        'balanced_global_marginals': balanced_global_marginals,  # NEW parameter
        'parallelize_training': parallelize_training,  # NEW parameter
        'NN_parameters': NN_parameters,
        'MIP_parameters': MIP_parameters,
        'RS_parameters':RS_parameters,
        'scaler': scaler,
        'init_bids_and_fitted_scaler': [None, None],
        'calc_efficiency_per_iteration': True,
        'isLegacy': False,
        'local_scaling_factor': 1.0,
        'res_path': res_path
    }

    # Run MLCA
    mlca_mechanism(**kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLCA')
    parser.add_argument('--domain', type=str, choices=['LSVM', 'SRVM', 'MRVM'])
    parser.add_argument('--q', type=float)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    main(domain=args.domain, seed=args.seed, q=args.q)
