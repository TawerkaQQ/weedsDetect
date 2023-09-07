import os
from tqdm import tqdm
import pickle
import argparse
import time
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from utils import set_seed, load_model, save, get_model, update_optimizer, get_data
from epoch import train_epoch, val_epoch, test_epoch
from cli import add_all_parsers

lossTrain =[]
lossVal =[]
accTrain =[]
accVal =[]
epoch_count = []


def train(args):
    set_seed(args, use_gpu=torch.cuda.is_available())
    train_loader, val_loader, test_loader, dataset_attributes = get_data(args.root, args.image_size, args.crop_size,
                                                                         args.batch_size, args.num_workers, args.pretrained)

    model = get_model(args, n_classes=dataset_attributes['n_classes'])
    criteria = CrossEntropyLoss()

    if args.use_gpu:
        print('USING GPU')
        torch.cuda.set_device(0)
        model.cuda()
        criteria.cuda()

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.mu, nesterov=True)

    # Containers for storing metrics over epochs
    loss_train, acc_train, topk_acc_train = [], [], []
    loss_val, acc_val, topk_acc_val, avgk_acc_val, class_acc_val = [], [], [], [], []

    save_name = args.save_name_xp.strip()
    save_dir = os.path.join(os.getcwd(), 'results', save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('args.k : ', args.k)

    lmbda_best_acc = None
    best_val_acc = float('-inf')

    for epoch in tqdm(range(args.n_epochs), desc='epoch', position=0):
        t = time.time()
        optimizer = update_optimizer(optimizer, lr_schedule=args.epoch_decay, epoch=epoch)

        loss_epoch_train, acc_epoch_train, topk_acc_epoch_train = train_epoch(model, optimizer, train_loader,
                                                                              criteria, loss_train, acc_train,
                                                                              topk_acc_train, args.k,
                                                                              dataset_attributes['n_train'],
                                                                              args.use_gpu)

        loss_epoch_val, acc_epoch_val, topk_acc_epoch_val, \
        avgk_acc_epoch_val, lmbda_val = val_epoch(model, val_loader, criteria,
                                                  loss_val, acc_val, topk_acc_val, avgk_acc_val,
                                                  class_acc_val, args.k, dataset_attributes, args.use_gpu)

        # save model at every epoch
        save(model, optimizer, epoch, os.path.join(save_dir, save_name + '_weights.tar'))

        # save model with best val accuracy
        if acc_epoch_val > best_val_acc:
            best_val_acc = acc_epoch_val
            lmbda_best_acc = lmbda_val
            save(model, optimizer, epoch, os.path.join(save_dir, save_name + '_weights_best_acc.tar'))

        print()
        print(f'epoch {epoch} took {time.time()-t:.2f}')
        print(f'loss_train : {loss_epoch_train}')
        print(f'loss_val : {loss_epoch_val}')
        print(f'acc_train : {acc_epoch_train} / topk_acc_train : {topk_acc_epoch_train}')
        print(f'acc_val : {acc_epoch_val} / topk_acc_val : {topk_acc_epoch_val} / '
              f'avgk_acc_val : {avgk_acc_epoch_val}')
        
        lossVal.append(loss_epoch_val)       
        lossTrain.append(loss_epoch_train)       
        accVal.append(acc_epoch_val)       
        accTrain.append(acc_epoch_train)       
        epoch_count.append(epoch)

        #Graph LossTrain/lossVal 
        
        fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
        ax.plot(epoch_count, lossTrain, label='lossTrain')  # Plot some data on the axes.
        ax.plot(epoch_count, lossVal, label='lossVal')  # Plot more data on the axes...

        ax.scatter(epoch_count, lossTrain)
        ax.scatter(epoch_count, lossVal)

        ax.set_xlabel('lossTrain')  # Add an x-label to the axes.
        ax.set_ylabel('lossVal')  # Add a y-label to the axes.
        ax.legend()  # Add a legend.
        plt.savefig('lossTrain_LossVal.png')

        #Graph accVal/lossVal 
        
        fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
        ax.plot(epoch_count, accVal, label='accVal')  # Plot some data on the axes.
        ax.plot(epoch_count, lossVal, label='lossVal')  # Plot more data on the axes...

        ax.scatter(epoch_count, accVal)
        ax.scatter(epoch_count, lossVal)

        ax.set_xlabel('accVal')  # Add an x-label to the axes.
        ax.set_ylabel('lossVal')  # Add a y-label to the axes.
        ax.legend()  # Add a legend.
        plt.savefig('accVal_lossVal.png')

        #Graph accTrain/LossTrain 
        
        fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
        ax.plot(epoch_count, accTrain, label='accTrain')  # Plot some data on the axes.
        ax.plot(epoch_count, lossTrain, label='lossTrain')  # Plot more data on the axes...

        ax.scatter(epoch_count, accTrain)
        ax.scatter(epoch_count, lossTrain)

        ax.set_xlabel('accTrain')  # Add an x-label to the axes.
        ax.set_ylabel('lossTrain')  # Add a y-label to the axes.
        ax.legend()  # Add a legend.
        plt.savefig('accTrain_lossTrain.png')


    # load weights corresponding to best val accuracy and evaluate on test
    load_model(model, os.path.join(save_dir, save_name + '_weights_best_acc.tar'), args.use_gpu)
    loss_test_ba, acc_test_ba, topk_acc_test_ba, \
    avgk_acc_test_ba, class_acc_test = test_epoch(model, test_loader, criteria, args.k,
                                                  lmbda_best_acc, args.use_gpu,
                                                  dataset_attributes)

    # Save the results as a dictionary and save it as a pickle file in desired location

    results = {'loss_train': loss_train, 'acc_train': acc_train, 'topk_acc_train': topk_acc_train,
               'loss_val': loss_val, 'acc_val': acc_val, 'topk_acc_val': topk_acc_val, 'class_acc_val': class_acc_val,
               'avgk_acc_val': avgk_acc_val,
               'test_results': {'loss': loss_test_ba,
                                'accuracy': acc_test_ba,
                                'topk_accuracy': topk_acc_test_ba,
                                'avgk_accuracy': avgk_acc_test_ba,
                                'class_acc_dict': class_acc_test},
               'params': args.__dict__}

    with open(os.path.join(save_dir, save_name + '.pkl'), 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_all_parsers(parser)
    args = parser.parse_args()
    train(args)
