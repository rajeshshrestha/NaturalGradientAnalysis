'''
Test the performance based on convergence and generalization
'''

'''Add path of other modules'''
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor import retrieve_and_process_data
from ml.trainer import train
import argparse
import numpy as np
from matplotlib import pyplot as plt

seed = np.random.randint(0, 10000)
np.random.seed(seed)

'''Parse arguments'''
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", choices=['separable_svm', 'overlapped_svm', 'weather', 'houseprice', 'cancer', 'ecoli'], required=True)
parser.add_argument("--test-proportion", default=0.2,
                    type=float, required=False)
parser.add_argument("--lr-decay", default=False, type=bool, required=False)
parser.add_argument("--lr-decay-type", default="smooth", type=str, choices=["smooth", "step"], required=False)
parser.add_argument("--sgd-lr", type=float, default=None)
parser.add_argument("--tengrad-lr", type=float, default=None)
parser.add_argument("--block-ngd-lr", type=float, default=None)
parser.add_argument("--exact-ngd-lr", type=float, default=None)
parser.add_argument("--layers", type=int, nargs="+",
                    required=False, default=[])
parser.add_argument("--activations", type=str, nargs="+", required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--model-type", default="dnn")
parser.add_argument("--lambda-reg", default=1e-6, type=float)
parser.add_argument("--running-loss-window", default=1, type=int)
parser.add_argument("--alpha", default=1e-6, type=float)



args = parser.parse_args()
print(args)

'''Retrieve data'''
train_data, test_data = retrieve_and_process_data(
    dataset=args.dataset, test_proportion=args.test_proportion)
feature_num, train_data_num, test_data_num = train_data.shape[0] - \
    1, train_data.shape[1], test_data.shape[1]

if args.task == "classification":
    fig_num = 4
else:
    fig_num = 3

plt.figure("training", figsize=(5, 10))

'''Train'''
if args.tengrad_lr:
    tengrad_grad_descent_train_info = train(layers=[feature_num, *args.layers, 1],
                                            activations=args.activations,
                                            train_data=train_data,
                                            test_data=test_data,
                                            update_type="tengrad",
                                            task=args.task,
                                            lr=args.tengrad_lr,
                                            epochs=args.epochs,
                                            mini_batch_size=args.batch_size,
                                            model=args.model_type,
                                            lambda_reg=args.lambda_reg,
                                            running_loss_window=args.running_loss_window,
                                            lr_decay=args.lr_decay,
                                            lr_decay_type = args.lr_decay_type,
                                            alpha=args.alpha)
    plt.subplot(fig_num, 1, 1)
    plt.plot(tengrad_grad_descent_train_info['iter_wise']['iterations'], np.log(
        tengrad_grad_descent_train_info['iter_wise']['running_loss']), '-', label=f'TENGRAD lr={args.tengrad_lr}')

    plt.subplot(fig_num, 1, 2)
    plt.plot(tengrad_grad_descent_train_info['epoch_wise']['epochs'], np.log(
        tengrad_grad_descent_train_info['epoch_wise']['loss']), '-', label=f'TENGRAD lr={args.tengrad_lr}')

    plt.subplot(fig_num, 1, 3)
    plt.plot(tengrad_grad_descent_train_info['epoch_wise']['epochs'], np.log(
        tengrad_grad_descent_train_info['epoch_wise']['val_loss']), '-', label=f'TENGRAD lr={args.tengrad_lr}')

    if args.task == "classification":
        plt.subplot(fig_num, 1, 4)
        plt.plot(tengrad_grad_descent_train_info['epoch_wise']['epochs'],
                 tengrad_grad_descent_train_info['epoch_wise']['val_accuracy'], '-', label=f'TENGRAD lr={args.tengrad_lr}')


if args.sgd_lr:
    sgd_descent_train_info = train(layers=[feature_num, *args.layers, 1],
                                   activations=args.activations,
                                   train_data=train_data,
                                   test_data=test_data,
                                   update_type="first_order",
                                   task=args.task,
                                   lr=args.sgd_lr,
                                   epochs=args.epochs,
                                   mini_batch_size=args.batch_size,
                                   model=args.model_type,
                                   lambda_reg=args.lambda_reg,
                                   running_loss_window=args.running_loss_window,
                                   lr_decay=args.lr_decay,
                                   lr_decay_type = args.lr_decay_type,
                                   alpha=args.alpha)
    plt.subplot(fig_num, 1, 1)
    plt.plot(sgd_descent_train_info['iter_wise']['iterations'], np.log(
        sgd_descent_train_info['iter_wise']['running_loss']), '--', label=f'SGD lr={args.sgd_lr}')

    plt.subplot(fig_num, 1, 2)
    plt.plot(sgd_descent_train_info['epoch_wise']['epochs'], np.log(
        sgd_descent_train_info['epoch_wise']['loss']), '--', label=f'SGD lr={args.sgd_lr}')

    plt.subplot(fig_num, 1, 3)
    plt.plot(sgd_descent_train_info['epoch_wise']['epochs'], np.log(
        sgd_descent_train_info['epoch_wise']['val_loss']), '--', label=f'SGD lr={args.sgd_lr}')

    if args.task == "classification":
        plt.subplot(fig_num, 1, 4)
        plt.plot(sgd_descent_train_info['epoch_wise']['epochs'],
                 sgd_descent_train_info['epoch_wise']['val_accuracy'], '--', label=f'SGD lr={args.sgd_lr}')

if args.block_ngd_lr:
    blockwise_ngd_descent_train_info = train(layers=[feature_num, *args.layers, 1],
                                             activations=args.activations,
                                             train_data=train_data,
                                             test_data=test_data,
                                             update_type="kfac",
                                             task=args.task,
                                             lr=args.block_ngd_lr,
                                             epochs=args.epochs,
                                             mini_batch_size=args.batch_size,
                                             model=args.model_type,
                                             lambda_reg=args.lambda_reg,
                                             running_loss_window=args.running_loss_window,
                                             lr_decay=args.lr_decay,
                                             lr_decay_type = args.lr_decay_type,
                                             alpha=args.alpha)
    plt.subplot(fig_num, 1, 1)
    plt.plot(blockwise_ngd_descent_train_info['iter_wise']['iterations'], np.log(
        blockwise_ngd_descent_train_info['iter_wise']['running_loss']), label=f'BlockWise NGD lr={args.block_ngd_lr}')

    plt.subplot(fig_num, 1, 2)
    plt.plot(blockwise_ngd_descent_train_info['epoch_wise']['epochs'], np.log(
        blockwise_ngd_descent_train_info['epoch_wise']['loss']), label=f'BlockWise NGD lr={args.block_ngd_lr}')

    plt.subplot(fig_num, 1, 3)
    plt.plot(blockwise_ngd_descent_train_info['epoch_wise']['epochs'], np.log(
        blockwise_ngd_descent_train_info['epoch_wise']['val_loss']), label=f'BlockWise NGD lr={args.block_ngd_lr}')

    if args.task == "classification":
        plt.subplot(fig_num, 1, 4)
        plt.plot(blockwise_ngd_descent_train_info['epoch_wise']['epochs'],
                 blockwise_ngd_descent_train_info['epoch_wise']['val_accuracy'], label=f'BlockWise NGD lr={args.block_ngd_lr}')

if args.exact_ngd_lr:
    ngd_train_info = train(layers=[feature_num, *args.layers, 1],
                           activations=args.activations,
                           train_data=train_data,
                           test_data=test_data,
                           update_type="natural_gradient",
                           task=args.task,
                           lr=args.exact_ngd_lr,
                           epochs=args.epochs,
                           mini_batch_size=args.batch_size,
                           model=args.model_type,
                           lambda_reg=args.lambda_reg,
                           running_loss_window=args.running_loss_window,
                           lr_decay=args.lr_decay,
                           lr_decay_type = args.lr_decay_type,
                           alpha=args.alpha)
    plt.subplot(fig_num, 1, 1)
    plt.plot(ngd_train_info['iter_wise']['iterations'], np.log(
        ngd_train_info['iter_wise']['running_loss']), label=f'Exact NGD lr={args.exact_ngd_lr}')

    plt.subplot(fig_num, 1, 2)
    plt.plot(ngd_train_info['epoch_wise']['epochs'], np.log(
        ngd_train_info['epoch_wise']['loss']), label=f'Exact NGD lr={args.exact_ngd_lr}')

    plt.subplot(fig_num, 1, 3)
    plt.plot(ngd_train_info['epoch_wise']['epochs'], np.log(
        ngd_train_info['epoch_wise']['val_loss']), label=f'Exact NGD lr={args.exact_ngd_lr}')

    if args.task == "classification":
        plt.subplot(fig_num, 1, 4)
        plt.plot(ngd_train_info['epoch_wise']['epochs'],
                 ngd_train_info['epoch_wise']['val_accuracy'], label=f'Exact NGD lr={args.exact_ngd_lr}')

    '''Plot train metrics'''
    plt.subplot(fig_num, 1, 1)
    plt.xlabel("Iterations")
    if args.running_loss_window > 1:
        plt.ylabel("Running Average of Log of Batch Loss")
    else:
        plt.ylabel("Log of Batch Loss")
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.subplot(fig_num, 1, 2)
    plt.xlabel("Epochs")
    plt.ylabel("Log of Average Train Loss")
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.subplot(fig_num, 1, 3)
    plt.xlabel("Epochs")
    plt.ylabel("Log of Test Loss")
    plt.legend(loc='upper right')
    plt.tight_layout()

    if args.task == "classification":
        plt.subplot(fig_num, 1, 4)
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy")
        plt.legend(loc='upper right')
        plt.tight_layout()
plt.show()
