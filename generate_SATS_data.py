import pickle
import logging
import numpy as np
import torch
from pysats import PySats


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def generate_data(SATS_domain,
                  bidder_id,
                  normalize,
                  seed,
                  num_train_data,
                  normalize_factor=1.0,
                  data_gen_method='random_uniform',
                  num_val_data=0,  # for methods == "random_uniform, admissible_random_uniform"
                  num_test_data=1000,  # for method == "random_uniform, admissible_random_uniform"
                  loadpath=None,  # only for method == "load"
                  val_ratio=0.2,  # only for method == "load"
                  *args, **kwargs):
    """
    Return train, val and test dataset splits for a single bidder of the bundle space.
    """

    # CREATE SATS INSTANCE FROM SATS_domain
    # --------------------------------------------------------------------------------------------------------------------------------
    if SATS_domain == 'GSVM':
        value_model = PySats.getInstance().create_gsvm(seed=seed)
        N = 7
        M = 18
    elif SATS_domain == 'LSVM':
        value_model = PySats.getInstance().create_lsvm(seed=seed)
        N = 6
        M = 18
    elif SATS_domain == 'SRVM':
        value_model = PySats.getInstance().create_srvm(seed=seed)
        N = 7
        M = 29
    elif SATS_domain == 'MRVM':
        value_model = PySats.getInstance().create_mrvm(seed=seed)
        N = 10
        M = 98
    else:
        raise NotImplementedError(f'Unknown SATS_domain:{SATS_domain}.')

    dataset_info = {'N': N,
                    'M': M,
                    'domain': SATS_domain,
                    'data_gen_method': data_gen_method,
                    'num_train_data': num_train_data,
                    'num_val_data': num_val_data,
                    'num_test_data': num_test_data,
                    'bidder_id': bidder_id}
    # --------------------------------------------------------------------------------------------------------------------------------

    # (1) METHOD: 'random_subset_path' SAMPLE ALONG A 1D-SUBSPACE WITH INCREASING BUNDLE SIZE FOR VISUALIZATION (only train and val)
    # --------------------------------------------------------------------------------------------------------------------------------
    if data_gen_method == 'random_subset_path':
        bundles = []
        iter = np.ones((1, M))
        bundles.append(iter.tolist()[0])
        for i in range(M):
            iter[0, np.random.choice(iter.nonzero()[1], 1)[0]] = 0
            bundles.append(iter.tolist()[0])
        bundles.reverse()
        X = np.array(bundles, dtype=np.float32)
        y = np.array(value_model.calculate_values(bidder_id, X), np.float32)

        # full bundle always in training set and null bundle never
        idx_tr = list(np.random.choice(list(range(1, M)), size=(num_train_data - 1), replace=False))
        idx_tr += [M]
        idx_test = [idx for idx in range(M) if idx not in idx_tr]

        X_train, y_train = X[idx_tr], y[idx_tr]
        X_test, y_test = X[idx_test], y[idx_test]

        if normalize:
            y_train_max = max(y_train)
            dataset_info['target_max'] = y_train_max
            y_train, y_test = y_train / y_train_max, y_test / y_train_max
        else:
            dataset_info['target_max'] = 1.0

        train = torch.utils.data.TensorDataset(torch.from_numpy(X_train),
                                               torch.from_numpy(y_train.reshape(-1, 1)))
        val = None

        test = torch.utils.data.TensorDataset(torch.from_numpy(X_test),
                                              torch.from_numpy(y_test.reshape(-1, 1)))
        return train, val, test, dataset_info
    # --------------------------------------------------------------------------------------------------------------------------------

    # (2) METHOD: 'admissible_random_uniform' SAMPLE UNIFORMLY FROM ADMISSIBLE BUNDLES VIA SATS
    # --------------------------------------------------------------------------------------------------------------------------------
    # 1. Ensure that FULL_BUNDLE always in training set, NULL_BUNDLE neither in training nor validation nor test set, UNIQUENESS, and that train/val/test are DISJOINT.
    # 2. Ensure that only bundles of interest are in train/val/test e.g. in GSVM regional bidder only gets items of size 4, national bidder only from national complement:
    #    use for this preimplemented sats uniform sampling methods
    # Remark: seeds are set in __main__ method
    elif data_gen_method == 'admissible_random_uniform':

        logging.info(f'Generate {data_gen_method} data')
        # Sampling method from SATS, which incorporates bidder specific restrictions:
        # e.g. in GSVM for regional bidders only bundles of up to size 4 are sampled and for national bidders only bundles that
        # contain items from the national-circle are sampled.
        # Remark: SATS does not ensure that bundles are unique, this needs to be taken care exogenously.
        # D = (X,y) in ({0,1}^m x R_+)^number_of_bids*(m+1)
        D = np.asarray(value_model.get_uniform_random_bids(bidder_id=bidder_id,
                                                           number_of_bids=(
                                                                                  num_train_data - 1) + num_val_data + num_test_data,
                                                           seed=seed), dtype=np.float32)

        # only use X from SATS generator, since then uniqueness check is easier
        X = D[:, :-1]

        full_bundle = np.array([1] * M, dtype=np.float32)
        empty_bundle = np.array([0] * M, dtype=np.float32)

        # Remove full bundle and null bundle if they were drawn
        full_idx = np.where(np.all(X == full_bundle, axis=1))[0]
        empty_idx = np.where(np.all(X == empty_bundle, axis=1))[0]
        if len(full_idx) > 0:
            X = np.delete(X, full_idx, axis=0)
        if len(empty_idx) > 0:
            X = np.delete(X, empty_idx, axis=0)
        #

        # UNIQUENESS
        X = np.unique(X, axis=0)
        seed_additional_bundle = None if seed is None else 10 ** 6 * seed
        while X.shape[0] != (num_train_data - 1) + num_val_data + num_test_data:
            # logging.debug(f'Generate new bundle: only {X.shape[0]+1} are unique but you asked for:{(num_train_data) + num_val_data + num_test_data}')
            dnew = np.asarray(
                value_model.get_uniform_random_bids(bidder_id=bidder_id, number_of_bids=1, seed=seed_additional_bundle))
            xnew = dnew[:, :-1]
            # logging.debug(xnew)
            # Check until new bundle is different from FULL_BUNDLE and NULL_BUNDLE
            while np.all(xnew == full_bundle) or np.all(xnew == empty_bundle):
                if seed_additional_bundle is not None: seed_additional_bundle += 1
                dnew = np.asarray(value_model.get_uniform_random_bids(bidder_id=bidder_id, number_of_bids=1,
                                                                      seed=seed_additional_bundle), dtype=np.float32)
                xnew = dnew[:, :-1]
                # logging.debug(f'RESAMPLE SINCE NULL OR FULL:{xnew}')
            X = np.concatenate((X, xnew), axis=0)
            X = np.unique(X, axis=0)
            if seed_additional_bundle is not None: seed_additional_bundle += 1
        #

        y = np.array(value_model.calculate_values(bidder_id, X), dtype=np.float32)
        X, y = unison_shuffled_copies(X, y)

        value_full_bundle = np.array([value_model.calculate_value(bidder_id, full_bundle)], dtype=np.float32)
        X_train = np.concatenate((X[:(num_train_data - 1)], np.array([1] * M, dtype=np.float32).reshape(1, -1)))
        y_train = np.concatenate((y[:(num_train_data - 1)], value_full_bundle))
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)

        X_val = X[(num_train_data - 1):(num_train_data - 1) + num_val_data]
        y_val = y[(num_train_data - 1):(num_train_data - 1) + num_val_data]
        X_val = X_val.astype(np.float32)
        y_val = y_val.astype(np.float32)

        X_test = X[(num_train_data - 1) + num_val_data:]
        y_test = y[(num_train_data - 1) + num_val_data:]
        X_test = X_test.astype(np.float32)
        y_test = y_test.astype(np.float32)

        logging.info('Check Uniqueness:')
        assert len(np.unique(np.concatenate((X_train, X_val, X_test)), axis=0)) == len(
            np.concatenate((X_train, X_val, X_test)))
        logging.info(
            f'IS UNIQUE: X in {np.concatenate((X_train, X_val, X_test), axis=0).shape}, y in {np.concatenate((y_train, y_val, y_test), axis=0).shape}')

        if normalize:
            y_train_max = max(y_train)
            dataset_info['target_max'] = y_train_max
            y_train, y_val, y_test = y_train / y_train_max, y_val / y_train_max, y_test / y_train_max
        else:
            dataset_info['target_max'] = 1.0

        train = torch.utils.data.TensorDataset(torch.from_numpy(X_train),
                                               torch.from_numpy(y_train.reshape(-1, 1)))
        val = torch.utils.data.TensorDataset(torch.from_numpy(X_val),
                                             torch.from_numpy(y_val.reshape(-1, 1)))
        test = torch.utils.data.TensorDataset(torch.from_numpy(X_test),
                                              torch.from_numpy(y_test.reshape(-1, 1)))
        return train, val, test, dataset_info
    # --------------------------------------------------------------------------------------------------------------------------------

    # (3) METHOD: ''random_uniform'' SAMPLE UNIFORMLY AT RANDOM FROM WHOLE BUNDLE SPACE IGNORING BIDDER SPECIFIC CONSTRAINTS FROM SATS
    # --------------------------------------------------------------------------------------------------------------------------------
    # 1. Ensure that FULL_BUNDLE always in training set, NULL_BUNDLE neither in training nor validation nor test set, UNIQUENESS, and that train/val/test are DISJOINT.
    # Remark: seeds are set in __main__ method
    elif data_gen_method == 'random_uniform':

        logging.info(f'Generate {data_gen_method} data')

        X = np.array(np.random.choice([0, 1],
                                      size=((num_train_data - 1) + num_val_data + num_test_data, M),
                                      replace=True), dtype=np.float32)

        full_bundle = np.array([1] * M, dtype=np.float32)
        empty_bundle = np.array([0] * M, dtype=np.float32)

        # Remove full bundle and null bundle if they were drawn
        full_idx = np.where(np.all(X == full_bundle, axis=1))[0]
        empty_idx = np.where(np.all(X == empty_bundle, axis=1))[0]
        if len(full_idx) > 0:
            X = np.delete(X, full_idx, axis=0)
        if len(empty_idx) > 0:
            X = np.delete(X, empty_idx, axis=0)
        #

        # UNIQUENESS
        X = np.unique(X, axis=0)
        while X.shape[0] != (num_train_data - 1) + num_val_data + num_test_data:
            # logging.debug(f'Generate new bundle since only {X.shape[0]+1} are unique but you asked for:{(num_train_data) + num_val_data + num_test_data}')
            xnew = np.array(np.random.choice([0, 1], size=(1, M), replace=True))
            # Check until new bundle is different from fulland null bundle
            while np.all(xnew == full_bundle) or np.all(xnew == empty_bundle):
                xnew = np.array(np.random.choice([0, 1], size=(1, M), replace=True, dtype=np.float32))
            X = np.concatenate((X, xnew), axis=0)
            X = np.unique(X, axis=0)
        #

        y = np.array(value_model.calculate_values(bidder_id, X), dtype=np.float32)
        X, y = unison_shuffled_copies(X, y)

        value_full_bundle = np.array([value_model.calculate_value(bidder_id, full_bundle)], dtype=np.float32)
        X_train = np.concatenate((X[:(num_train_data - 1)], np.array([1] * M, dtype=np.float32).reshape(1, -1)))
        y_train = np.concatenate((y[:(num_train_data - 1)], value_full_bundle))
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)

        X_val = X[(num_train_data - 1):(num_train_data - 1) + num_val_data]
        y_val = y[(num_train_data - 1):(num_train_data - 1) + num_val_data]
        X_val = X_val.astype(np.float32)
        y_val = y_val.astype(np.float32)

        X_test = X[(num_train_data - 1) + num_val_data:]
        y_test = y[(num_train_data - 1) + num_val_data:]
        X_test = X_test.astype(np.float32)
        y_test = y_test.astype(np.float32)

        logging.info('Check Uniqueness')
        assert len(np.unique(np.concatenate((X_train, X_val, X_test)), axis=0)) == len(
            np.concatenate((X_train, X_val, X_test)))
        logging.info(
            f'IS UNIQUE: X in {np.concatenate((X_train, X_val, X_test), axis=0).shape}, y in {np.concatenate((y_train, y_val, y_test), axis=0).shape}')

        if normalize:
            y_train_max = max(y_train)
            dataset_info['target_max'] = y_train_max
            y_train, y_val, y_test = y_train / y_train_max, y_val / y_train_max, y_test / y_train_max
        else:
            dataset_info['target_max'] = 1.0

        train = torch.utils.data.TensorDataset(torch.from_numpy(X_train),
                                               torch.from_numpy(y_train.reshape(-1, 1)))
        val = torch.utils.data.TensorDataset(torch.from_numpy(X_val),
                                             torch.from_numpy(y_val.reshape(-1, 1)))
        test = torch.utils.data.TensorDataset(torch.from_numpy(X_test),
                                              torch.from_numpy(y_test.reshape(-1, 1)))
        return train, val, test, dataset_info
    # --------------------------------------------------------------------------------------------------------------------------------

    # (4) METHOD: 'load' LOAD A PREPARED DATASET
    # --------------------------------------------------------------------------------------------------------------------------------
    # Remark: seeds are set in __main__ method
    elif data_gen_method == 'load':
        dataset = pickle.load(open(loadpath, 'rb'))
        X, y = dataset[:, :M], dataset[:, M + bidder_id]
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        X, y = unison_shuffled_copies(X, y)

        X_train, y_train = X[:num_train_data], \
                           y[:num_train_data]
        X_val, y_val = X[num_train_data:int(len(X) * 0.2 + num_train_data)], \
                       y[num_train_data:int(len(X) * 0.2 + num_train_data)]
        X_test, y_test = X[int(len(X) * 0.2 + num_train_data):], \
                         y[int(len(X) * 0.2 + num_train_data):]
        if normalize:
            y_train_max = max(y_train) * (1 / normalize_factor)
            dataset_info['target_max'] = y_train_max
            y_train, y_val, y_test = y_train / y_train_max, y_val / y_train_max, y_test / y_train_max
        else:
            dataset_info['target_max'] = 1.0

        train = torch.utils.data.TensorDataset(torch.from_numpy(X_train),
                                               torch.from_numpy(y_train.reshape(-1, 1)))
        val = torch.utils.data.TensorDataset(torch.from_numpy(X_val),
                                             torch.from_numpy(y_val.reshape(-1, 1)))
        test = torch.utils.data.TensorDataset(torch.from_numpy(X_test),
                                              torch.from_numpy(y_test.reshape(-1, 1)))
        return train, val, test, dataset_info
    # --------------------------------------------------------------------------------------------------------------------------------

    else:
        raise NotImplementedError(f'Data Generation method:{data_gen_method} not implemented!')
    # --------------------------------------------------------------------------------------------------------------------------------
