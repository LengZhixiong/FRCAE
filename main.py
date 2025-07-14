import argparse
import random
import numpy as np
import torch
import os
from data_generator import miniImagenet
from maml import MAML
import csv
import pandas as pd
from collections import Counter
from numpy.linalg import norm
from torch.optim.lr_scheduler import StepLR
import pickle
import json
import sys
import time
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Meta')
parser.add_argument('--datasource', default='cifar', type=str,
                    help='cifar or miniimagenet')
parser.add_argument('--num_classes', default=5, type=int,
                    help='number of classes used in classification (e.g. 5-way classification).')
parser.add_argument('--num_test_task', default=600, type=int, help='number of test tasks.')
parser.add_argument('--test_epoch', default=-1, type=int, help='test epoch, only work when test start')

# Training options
parser.add_argument('--meta_batch_size', default=4, type=int, help='number of tasks sampled per meta-update')
parser.add_argument('--update_lr', default=0.01, type=float, help='inner learning rate')
parser.add_argument('--meta_lr', default=0.001, type=float, help='the base learning rate of the generator')
parser.add_argument('--num_updates', default=5, type=int, help='num_updates in maml')
parser.add_argument('--num_updates_test', default=10, type=int, help='num_updates in maml')
parser.add_argument('--update_batch_size', default=5, type=int,
                    help='number of examples used for inner gradient update (K for K-shot learning). train')
parser.add_argument('--update_batch_size_eval', default=15, type=int,
                    help='number of examples used for inner gradient test (K for K-shot learning).')
parser.add_argument('--num_filters', default=64, type=int,
                    help='number of filters for conv nets -- 32 for miniimagenet, cifar.')
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')

# Logging, saving, and testing options
parser.add_argument('--learner', default='anil', type=str)
parser.add_argument('--logdir', default='./train_logs', type=str,
                    help='directory for summaries and checkpoints.')
parser.add_argument('--datadir', default='/dataset_dir', type=str, help='directory for datasets.')

parser.add_argument('--resume', default=0, type=int, help='resume training if there is a model available')
parser.add_argument('--train', default=True, type=bool, help='True to train, False to test.')
parser.add_argument('--test_set', default=1, type=int,
                    help='Set to true to test on the the test set, False for the validation set.')

parser.add_argument("--seed", default=1)

parser.add_argument("--break_iter", type=int, default=30000)

parser.add_argument("--kl", type=int, default=1)

parser.add_argument("--cae", type=int, default=1)

parser.add_argument("--dropout", type=float, default=0)

parser.add_argument('--trans_data', default='aircraft', type=str,
                    help='cifar or miniimagenet')

parser.add_argument('--transfer', type=int, default=0)

args = parser.parse_args()


if args.datasource == 'cifar':
    fully_connected = 256
else:
    fully_connected = 1600
    args.break_iter = 60000

if args.update_batch_size != 1:
    if args.break_iter == 60000:
        args.break_iter = 30000
    else:
        args.break_iter = 15000

print(args)

def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(args.seed)
    
exp_string = f'kl{args.kl}_cae{args.cae}_seed{args.seed}_{args.datasource}_{args.learner}_{args.num_classes}way-{args.update_batch_size}shot_{args.meta_batch_size}batch_{args.num_filters}filters'

if args.num_updates != 5:
    exp_string += '_' + str(args.num_updates) + 'inner'

print(exp_string)

def save_json(args):
    data = vars(args)
    json_data = json.dumps(data, indent=None)
    json_data = json_data.replace(', ', '\n')
    json_data = json_data.replace('{', '')
    json_data = json_data.replace('}', '')
    with open('{0}/{1}/parameter.json'.format(args.logdir, exp_string), 'w') as f:
        f.write(json_data)

def save_statistics(line_to_add, filename="train_summary.csv", create=False): 
    summary_filename = "{0}/{1}/{2}".format(args.logdir, exp_string, filename)
    if create and not os.path.exists(summary_filename): 
        with open(summary_filename, 'w') as f: 
            writer = csv.writer(f) 
            writer.writerow(line_to_add) 
    else: 
        with open(summary_filename, 'a') as f: 
            writer = csv.writer(f) 
            writer.writerow(line_to_add) 

def read_statistics(filename="val_summary.csv", top_n_models=10, step = 0): 
    summary_filename = "{0}/{1}/{2}".format(args.logdir, exp_string, filename)
    df = pd.read_csv(summary_filename,encoding='GB18030')

    val_acc = np.copy(df['acc'])
    sorted_idx = np.argsort(val_acc, axis=0).astype(dtype=np.int32)[::-1][:top_n_models]
    top_val_epoch_idx = df['val_epoch'].loc[sorted_idx].values

    return top_val_epoch_idx

def train(args, maml, optimiser, data_train, data_val):
    Print_Iter = 100
    Save_Iter = 500
    print_loss, print_acc, print_loss_support = 0.0, 0.0, 0.0

    dataloader = miniImagenet(args, 'train', data_train)

    dropout = 0

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataloader):
        maml.train()

        if step > args.break_iter:
            break

        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).cuda(), y_spt.squeeze(0).cuda(), \
                             x_qry.squeeze(0).cuda(), y_qry.squeeze(0).cuda()
        task_losses = []
        task_acc = []
        kl_loss = []

        for meta_batch in range(args.meta_batch_size):
            if args.learner == 'anil':
                loss_val, acc_val, logits = maml.forward_anil(x_spt[meta_batch], y_spt[meta_batch], x_qry[meta_batch],
                                                      y_qry[meta_batch], dropout)
            else:
                loss_val, acc_val, logits = maml.forward_maml(x_spt[meta_batch], y_spt[meta_batch], x_qry[meta_batch],
                                                  y_qry[meta_batch], dropout)
            task_losses.append(loss_val)
            task_acc.append(acc_val)
            kl_loss.append(F.kl_div(logits[1].softmax(dim=-1).log(), logits[0].softmax(dim=-1), reduction='batchmean'))

        meta_batch_loss = torch.stack(task_losses).mean()

        if args.kl != 0:
            meta_batch_loss += torch.stack(kl_loss).mean()

        meta_batch_acc = torch.stack(task_acc).mean()

        optimiser.zero_grad()
        meta_batch_loss.backward()
        optimiser.step()

        t_error_var = torch.stack(task_losses).detach().cpu().numpy().var()
        t_error = list(torch.stack(task_losses).detach().cpu().numpy())
        t_error.append(t_error_var)

        save_statistics(t_error, filename="loss_summary.csv")

        if step != 0 and step % Print_Iter == 0:
            task_losses_var = []
            print('iter: {}, loss_all: {}, acc: {}'.format(step, print_loss, print_acc))
            save_statistics([step, print_loss.item(), print_acc.item()])
            
            print_loss, print_acc = 0.0, 0.0
        else:
            print_loss += meta_batch_loss / Print_Iter
            print_acc += meta_batch_acc / Print_Iter

        if step != 0 and step % Save_Iter == 0:
            val(args, maml, step, data_val)
            torch.save(maml.state_dict(), '{0}/{2}/model{1}'.format(args.logdir, step, exp_string))

def val(args, maml, val_epoch, data_val):
    res_acc = []
    task_losses = []
    args.train = False
    meta_batch_size = args.meta_batch_size
    args.meta_batch_size = 1

    dataloader_test = miniImagenet(args, 'val', data_val)
    maml.eval()
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataloader_test):
        if step > args.num_test_task:
            break
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).cuda(), y_spt.squeeze(0).cuda(), \
                                     x_qry.squeeze(0).cuda(), y_qry.squeeze(0).cuda()
        if args.learner == 'anil':
            loss_val, acc_val, logits = maml.forward_anil(x_spt, y_spt, x_qry, y_qry)
        else:
            loss_val, acc_val, logits = maml.forward_maml(x_spt, y_spt, x_qry, y_qry)
        res_acc.append(acc_val.item())
        task_losses.append(loss_val.item())
        
    res_acc = np.array(res_acc)
    task_losses = np.array(task_losses)

    print('val_epoch is {}, loss is {}, acc is {}, ci95 is {}'.format(val_epoch, np.mean(task_losses), np.mean(res_acc),
                                                           1.96 * np.std(res_acc) / np.sqrt(
                                                               args.num_test_task * args.meta_batch_size)))

    save_statistics([val_epoch, np.mean(task_losses), np.mean(res_acc), 1.96 * np.std(res_acc) / np.sqrt(
                                            args.num_test_task * args.meta_batch_size)], filename="val_summary.csv")

    args.train = True
    args.meta_batch_size = meta_batch_size

def test(args, maml, data_test):
    res_acc = []
    args.train = False
    meta_batch_size = args.meta_batch_size
    args.meta_batch_size = 1

    dataloader_test = miniImagenet(args, 'test', data_test)
    maml.eval()
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataloader_test):
        if step > args.num_test_task:
            break
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).cuda(), y_spt.squeeze(0).cuda(), \
                                     x_qry.squeeze(0).cuda(), y_qry.squeeze(0).cuda()
        if args.learner == 'anil':
            _, acc_val, logits = maml.forward_anil(x_spt, y_spt, x_qry, y_qry)
        else:
            _, acc_val, logits = maml.forward_maml(x_spt, y_spt, x_qry, y_qry)
        res_acc.append(acc_val.item())

    res_acc = np.array(res_acc)

    print('acc is {}, ci95 is {}'.format(np.mean(res_acc),
                                         1.96 * np.std(res_acc) / np.sqrt(args.num_test_task * args.meta_batch_size)))
    args.train = 1
    args.meta_batch_size = meta_batch_size

    return np.mean(res_acc), 1.96 * np.std(res_acc) / np.sqrt(args.num_test_task * 1)

def main():
    maml = MAML(args, fully_connected = fully_connected).cuda()

    if args.datasource == 'miniimagenet':
        data_file = '{0}/{1}/miniimagenet_train.pkl'.format(args.datadir, args.datasource)
        data_train = pickle.load(open(data_file, 'rb'))
        data_train = torch.tensor(np.transpose(data_train, (0, 1, 4, 2, 3)))

        data_file = '{0}/{1}/miniimagenet_val.pkl'.format(args.datadir, args.datasource)
        data_val = pickle.load(open(data_file, 'rb'))
        data_val = torch.tensor(np.transpose(data_val, (0, 1, 4, 2, 3)))

        data_file = '{0}/{1}/miniimagenet_test.pkl'.format(args.datadir, args.datasource)
        data_test = pickle.load(open(data_file, 'rb'))
        data_test = torch.tensor(np.transpose(data_test, (0, 1, 4, 2, 3)))
    elif args.datasource == 'cifar':
        data_file = '{0}/{1}/cifar_train.pkl'.format(args.datadir, args.datasource)
        data_train = pickle.load(open(data_file, 'rb'))
        data_train = torch.tensor(np.transpose(data_train, (0, 1, 4, 2, 3)))

        data_file = '{0}/{1}/cifar_val.pkl'.format(args.datadir, args.datasource)
        data_val = pickle.load(open(data_file, 'rb'))
        data_val = torch.tensor(np.transpose(data_val, (0, 1, 4, 2, 3)))

        data_file = '{0}/{1}/cifar_test.pkl'.format(args.datadir, args.datasource)
        data_test = pickle.load(open(data_file, 'rb'))
        data_test = torch.tensor(np.transpose(data_test, (0, 1, 4, 2, 3)))
    elif args.datasource == 'dogs':
        data_file = '{0}/{1}/dogs_train.pkl'.format(args.datadir, args.datasource)
        data_train = pickle.load(open(data_file, 'rb'))
        data_train = torch.tensor(np.transpose(data_train, (0, 1, 4, 2, 3)))

        data_file = '{0}/{1}/dogs_val.pkl'.format(args.datadir, args.datasource)
        data_val = pickle.load(open(data_file, 'rb'))
        data_val = torch.tensor(np.transpose(data_val, (0, 1, 4, 2, 3)))

        data_file = '{0}/{1}/dogs_test.pkl'.format(args.datadir, args.datasource)
        data_test = pickle.load(open(data_file, 'rb'))
        data_test = torch.tensor(np.transpose(data_test, (0, 1, 4, 2, 3)))
    elif args.datasource == 'dtd':
        data_file = '{0}/{1}/dtd_train.pkl'.format(args.datadir, args.datasource)
        data_train = pickle.load(open(data_file, 'rb'))
        data_train = torch.tensor(np.transpose(data_train, (0, 1, 4, 2, 3)))

        data_file = '{0}/{1}/dtd_val.pkl'.format(args.datadir, args.datasource)
        data_val = pickle.load(open(data_file, 'rb'))
        data_val = torch.tensor(np.transpose(data_val, (0, 1, 4, 2, 3)))

        data_file = '{0}/{1}/dtd_test.pkl'.format(args.datadir, args.datasource)
        data_test = pickle.load(open(data_file, 'rb'))
        data_test = torch.tensor(np.transpose(data_test, (0, 1, 4, 2, 3)))

    if not os.path.exists(args.logdir + '/' + exp_string + '/'):
        os.makedirs(args.logdir + '/' + exp_string + '/')
    
    save_json(args)

    meta_optimiser = torch.optim.Adam(list(maml.parameters()),
                                      lr=args.meta_lr, weight_decay=args.weight_decay)

    if args.train == 1:

        if not os.path.exists("{0}/{1}/train_summary.csv".format(args.logdir, exp_string)):
            train_summary = ["train_iter", "loss", "acc"]
            save_statistics(train_summary, create=True)

        if not os.path.exists("{0}/{1}/val_summary.csv".format(args.logdir, exp_string)):    
            val_summary = ["val_epoch", "loss", "acc", "ci95"]   
            save_statistics(val_summary, filename="val_summary.csv", create=True)

        if not os.path.exists("{0}/{1}/loss_summary.csv".format(args.logdir, exp_string)):    
            loss_summary = ["loss0", "loss1", "loss2", "loss3", "loss_var"]   
            save_statistics(loss_summary, filename="loss_summary.csv", create=True)

        train(args, maml, meta_optimiser, data_train, data_val)

        args.train = 0

        test_acc = []
        test_ci95 = []
        test_summary = ["acc", "ci95"]
        filename = '{0}_test_summary.csv'.format(exp_string)
        save_statistics(test_summary, filename=filename, create=True)
        top_val_epoch_idx = read_statistics()

        for i in range(len(top_val_epoch_idx)):
            model_file = '{0}/{1}/model{2}'.format(args.logdir, exp_string, top_val_epoch_idx[i])
            maml.load_state_dict(torch.load(model_file))
            acc, ci95 = test(args, maml, data_test)
            test_acc.append(acc)
            test_ci95.append(ci95)
            save_statistics([acc, ci95], filename=filename)
        save_statistics(["acc_mean", "ci95_mean"], filename=filename)
        save_statistics([np.mean(test_acc), np.mean(test_ci95)], filename=filename)
        print('acc_mean is {}, ci95_mean is {}'.format(np.mean(test_acc), np.mean(test_ci95)))
    else:
        if args.datasource == 'miniimagenet':
            data_file = '{0}/{1}/miniimagenet_test.pkl'.format(args.datadir, args.datasource)
            data_test = pickle.load(open(data_file, 'rb'))
            data_test = torch.tensor(np.transpose(data_test, (0, 1, 4, 2, 3)))
        else:
            data_file = '{0}/{1}/cifar_test.pkl'.format(args.datadir, args.datasource)
            data_test = pickle.load(open(data_file, 'rb'))
            data_test = torch.tensor(np.transpose(data_test, (0, 1, 4, 2, 3)))

        if args.trans_data == 'dogs':
            data_file = '{0}/{1}/dogs_test.pkl'.format(args.datadir, args.trans_data)
            data_test = pickle.load(open(data_file, 'rb'))
            data_test = torch.tensor(np.transpose(data_test, (0, 1, 4, 2, 3)))
        elif args.trans_data == 'dtd':
            data_file = '{0}/{1}/dtd_test.pkl'.format(args.datadir, args.trans_data)
            data_test = pickle.load(open(data_file, 'rb'))
            data_test = torch.tensor(np.transpose(data_test, (0, 1, 4, 2, 3)))

        test_acc = []
        test_ci95 = []
        test_summary = ["acc", "ci95"]
        if args.transfer != 0:
            filename = '{0}_{1}_test_summary.csv'.format(exp_string, args.trans_data)
        else:
            filename = '{0}_new_test_summary.csv'.format(exp_string)
        save_statistics(test_summary, filename=filename, create=True)
        top_val_epoch_idx = read_statistics()
        print(top_val_epoch_idx)
        for i in range(len(top_val_epoch_idx)):
            model_file = '{0}/{1}/model{2}'.format(args.logdir, exp_string, top_val_epoch_idx[i])
            maml.load_state_dict(torch.load(model_file))
            acc, ci95 = test(args, maml, data_test)
            test_acc.append(acc)
            test_ci95.append(ci95)
            save_statistics([acc, ci95], filename=filename)
        save_statistics(["acc_mean", "ci95_mean"], filename=filename)
        save_statistics([np.mean(test_acc), np.mean(test_ci95)], filename=filename)
        print('acc_mean is {}, ci95_mean is {}'.format(np.mean(test_acc), np.mean(test_ci95)))

if __name__ == '__main__':
    main()

