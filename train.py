import argparse
import datetime
import logging
import os
import time
import numpy as np
import requests
import torch

from utils.data import SampleGenerator
from utils.utils import setSeed, initLogging, loadData

def loadEngine(configuration):
    if configuration['alias'] == 'FedDAD':
        from model.model import FedDADEngine
        load_engine = FedDADEngine(configuration)

    return load_engine


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--alias', type=str, default='FedDAD')
    parser.add_argument('--dataset', type=str, default='movielens')
    parser.add_argument('--data_file', type=str, default='ml-100k.dat')
    parser.add_argument('--model_dir', type=str, default='results/checkpoints/{}/{}/[{}]Epoch{}.model')
    parser.add_argument('--clients_sample_ratio', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--decay_rate', type=float, default=0.97)
    parser.add_argument('--num_round', type=int, default=100)
    parser.add_argument('--local_epoch', type=int, default=10)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr_u_p', type=float, default=5e-1)
    parser.add_argument('--lr_u_c', type=float, default=5e-1)
    parser.add_argument('--lr_i_p', type=float, default=2e2)
    parser.add_argument('--lr_i_c', type=float, default=2e2)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--l2_regularization', type=float, default=1e-4)
    parser.add_argument('--num_negative', type=int, default=4)
    parser.add_argument('--lambda', type=float, default=0.1)
    parser.add_argument('--mu', type=float, default=0.1)
    parser.add_argument('--regular', type=str, default='l1')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--type', type=str, default='seed')
    parser.add_argument('--comment', type=str, default='default')
    parser.add_argument('--on_server', type=bool, default=False)
    parser.add_argument('--vary_param', type=str, default='tanh')

    args = parser.parse_args()

    # Config
    config = vars(args)

    device = torch.device("cuda:{}".format(config['device_id']) if torch.cuda.is_available() and config['use_cuda'] else "cpu")
    torch.cuda.set_device(config['device_id'])

    # Set random seed
    setSeed(config['seed'])

    # Set cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Logging.
    path = 'logs/'
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    log_file_name = os.path.join(path,
                                 '[{}]-[{}.{}]-[{}].txt'.format(config['alias'], config['dataset'],
                                                                config['data_file'].split('.')[0], current_time))
    initLogging(log_file_name)

    # Load Data
    ratings, config['num_users'], config['num_items'] = loadData('../datasets', config['dataset'], config['data_file'])

    # create folder to save checkpoints
    checkpoint_path = 'results/checkpoints/{}/{}/'.format(config['alias'], config['dataset'])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    engine = loadEngine(config)

    logging.info(str(config))

    # DataLoader for training
    sample_generator = SampleGenerator(ratings=ratings)
    validate_data = sample_generator.validate_data
    test_data = sample_generator.test_data

    # Initialize for training
    test_hrs, test_ndcgs, val_hrs, val_ndcgs, train_losses = [], [], [], [], []
    best_test_hr, final_test_round = 0, 0
    sparsity_user, sparsity_item = [], []
    avg_sparsity = []########

    user_commonality = torch.nn.Embedding(num_embeddings=config['num_users'], embedding_dim=config['latent_dim']).to(device)
    item_commonality = torch.nn.Embedding(num_embeddings=config['num_items'], embedding_dim=config['latent_dim']).to(device)

    times = []

    for iteration in range(config['num_round']):

        logging.info('--------------- Round {} starts ! ---------------'.format(iteration + 1))

        if config['alias'] != 'NCF':
            train_data = sample_generator.store_all_train_data(config['num_negative'])
        else:
            train_data = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])

        # 1. Train Phase
        start_time = time.perf_counter()
        
        train_loss, sparse_user_value, sparse_item_value = engine.federatedTrainOneRound(train_data, user_commonality, item_commonality, iteration)
        end_time = time.perf_counter()

        times.append((end_time - start_time))

        logging.info('[{}/{}][{}] Time consuming: {:.4f}'.format(config['dataset'],
                                                                 config['data_file'],
                                                                 config['alias'],
                                                                 (end_time - start_time)))

        loss = sum(train_loss.values()) / len(train_loss.keys())
        train_losses.append(loss)
        sparsity_user.append(sparse_user_value)
        sparsity_item.append(sparse_item_value)
        ##############
        avg_sparse = (sparse_user_value + sparse_item_value) / 2
        avg_sparsity.append(avg_sparse)

        logging.info(
            '[Epoch {}/{}][Train] Loss = {:.4f}, User Sparsity = {:.4f}, Item Sparsity = {:.4f}, Avg Sparsity = {:.4f}'.format(iteration + 1, config['num_round'], loss,
                                                                           sparse_user_value, sparse_item_value, avg_sparse))

        # 2. Evaluations on Test set
        hr, ndcg = engine.federatedEvaluate(test_data)

        logging.info(
            '[Epoch {}/{}][Test] HR@{} = {:.4f}, NDCG@{} = {:.4f}'.format(iteration + 1, config['num_round'],
                                                                          config['top_k'], hr, config['top_k'], ndcg))

        test_hrs.append(hr)
        test_ndcgs.append(ndcg)

        # Choose the model has the best performances
        if hr >= best_test_hr:
            best_test_hr = hr
            final_test_round = iteration

        # 3. Evaluations on Validation set
        val_hr, val_ndcg = engine.federatedEvaluate(validate_data)

        logging.info(
            '[Epoch {}/{}][Validation] HR@{} = {:.4f}, NDCG@{} = {:.4f}'.format(iteration + 1, config['num_round'],
                                                                                config['top_k'], val_hr,
                                                                                config['top_k'],
                                                                                val_ndcg))
        val_hrs.append(val_hr)
        val_ndcgs.append(val_ndcg)

    logging.info('--------------- The model training is finished ---------------')

    logging.info('[{}/{}][{}] Time consuming: {:.4f}'.format(config['dataset'],
                                                             config['data_file'],
                                                             config['alias'],
                                                             sum(times)))

    content = config.copy()

    # delete some unuseful key-value
    del content['device_id']
    del content['use_cuda']
    del content['model_dir']

    logging.info(str(content))

    # add some useful key-value
    content['finish_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    content['hr'] = val_hrs[final_test_round]
    content['ndcg'] = val_ndcgs[final_test_round]

    # save useful data
    save_path = 'DADresults/{}/{}/{}'.format(content['alias'], content['dataset'], content['type'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    #
    result_file = os.path.join(save_path, '{}.{}.txt'.format(config['dataset'], os.path.basename(config['data_file']).split('.')[0]))
    result_dir = os.path.dirname(result_file)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(result_file, 'a') as file:
        file.write(str(content) + '\n')

    # 修改此处
    data_file = os.path.join(save_path, '[{}]-[{}-{:.2e}-{:.2e}]-[HR{:.4f}-NDCG{:.4f}]-[{}].npz'.format(
        os.path.basename(content['data_file']).split('.')[0],
        config['regular'], content['lambda'],
        content['mu'], content['hr'],
        content['ndcg'],
        content['comment']
    ))

    data_dir = os.path.dirname(data_file)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    np.savez(data_file, test_hrs=test_hrs, test_ndcgs=test_ndcgs, val_hrs=val_hrs, val_ndcgs=val_ndcgs,
             train_losses=train_losses, sparsity_user=sparsity_user, sparsity_item=sparsity_item, avg_sparsity=avg_sparsity)

    logging.info('hit_list: {}'.format(test_hrs))
    logging.info('ndcg_list: {}'.format(test_ndcgs))
    logging.info('avg_sparsity_list: {}'.format(avg_sparsity))
    notice = 'DAD in {} Best test hr: {:.4f}, ndcg: {:.4f}, avg sparsity: {:.4f} at round {}'.format(config['dataset'],
                                                                                                     test_hrs[final_test_round],
                                                                                                     test_ndcgs[final_test_round],
                                                                                                     avg_sparsity[final_test_round],
                                                                                                     final_test_round + 1)

    logging.info(notice)
