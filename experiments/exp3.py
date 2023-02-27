'''
Test the performance based on convergence and generalization
'''

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

'''Add path of other modules'''
from matplotlib import pyplot as plt
import numpy as np
import argparse
from ml.trainer import train
from data_processor import retrieve_and_process_data
from time import time
from memory_profiler import memory_usage

seed = np.random.randint(0, 10000)
np.random.seed(seed)

'''Parse arguments'''
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", choices=['separable_svm', 'overlapped_svm', 'weather', 'houseprice', 'cancer', 'ecoli'], required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--model-type", default="dnn")
parser.add_argument("--use-tengrad", default=False, action='store_true')
parser.add_argument("--use-block-ngd", default=False, action='store_true')
parser.add_argument("--use-exact-ngd", default=False, action='store_true')
parser.add_argument("--analyze-time", default=False, action='store_true')
parser.add_argument("--analyze-memory", default=False, action='store_true')
parser.add_argument("--use-sgd", default=False, action='store_true')
parser.add_argument("--hidden-layer-size", default=4, type=int)
parser.add_argument("--hidden-layer-nums", default=[4], nargs="+", type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    '''Retrieve data'''
    train_data, test_data = retrieve_and_process_data(
        dataset=args.dataset, test_proportion=0.2)

    train_data, test_data = train_data[:,:args.batch_size], test_data[:, :args.batch_size]
    feature_num, train_data_num, test_data_num = train_data.shape[0] - \
        1, train_data.shape[1], test_data.shape[1]


    lr=1e-12

    memory_time_info = {'hidden-layer-size': args.hidden_layer_size,
                        'tengrad': {'hidden-layer-num':[],'time':[], 'avg-space':[], 'min-space':[], 'max-space':[]},
                        'block-ngd':{'hidden-layer-num':[],'time':[], 'avg-space':[], 'min-space':[], 'max-space':[]},
                        'exact-ngd':{'hidden-layer-num':[],'time':[], 'avg-space':[], 'min-space':[], 'max-space':[]},
                        'sgd':{'hidden-layer-num':[],'time':[], 'avg-space':[], 'min-space':[], 'max-space':[]}}

    for hidden_layer_num in args.hidden_layer_nums:
        if args.use_tengrad:
            '''Run tengrad'''
            start_time = time()
            f = lambda : train(layers=[feature_num, *([args.hidden_layer_size]*hidden_layer_num), 1],
                                                    activations=['sigmoid' for i in range(hidden_layer_num+1)],
                                                    train_data=train_data,
                                                    test_data=test_data,
                                                    update_type="tengrad",
                                                    task=args.task,
                                                    lr=lr,
                                                    epochs=args.epochs,
                                                    mini_batch_size=args.batch_size,
                                                    model=args.model_type,
                                                    )
            tengrad_memory_info = memory_usage(f, interval=1e-3)
            end_time = time()
            time_taken = end_time-start_time

            memory_time_info['tengrad']['hidden-layer-num'].append(hidden_layer_num)
            memory_time_info['tengrad']['time'].append(time_taken)
            memory_time_info['tengrad']['avg-space'].append(np.mean(tengrad_memory_info))
            memory_time_info['tengrad']['min-space'].append(np.min(tengrad_memory_info))
            memory_time_info['tengrad']['max-space'].append(np.max(tengrad_memory_info))


        if args.use_sgd:
            start_time = time()
            f = lambda : train(layers=[feature_num, *([args.hidden_layer_size]*hidden_layer_num), 1],
                                            activations=['sigmoid' for i in range(hidden_layer_num+1)],
                                            train_data=train_data,
                                            test_data=test_data,
                                            update_type="first_order",
                                            task=args.task,
                                            lr=lr,
                                            epochs=args.epochs,
                                            mini_batch_size=args.batch_size,
                                            model=args.model_type,
                                            )
            sgd_memory_info = memory_usage(f, interval=1e-3)
            end_time = time()
            time_taken = end_time-start_time

            memory_time_info['sgd']['hidden-layer-num'].append(hidden_layer_num)
            memory_time_info['sgd']['time'].append(time_taken)
            memory_time_info['sgd']['avg-space'].append(np.mean(sgd_memory_info))
            memory_time_info['sgd']['min-space'].append(np.min(sgd_memory_info))
            memory_time_info['sgd']['max-space'].append(np.max(sgd_memory_info))

        if args.use_block_ngd:
            start_time = time()
            f = lambda : train(layers=[feature_num, *([args.hidden_layer_size]*hidden_layer_num), 1],
                                                    activations=['sigmoid' for i in range(hidden_layer_num+1)],
                                                    train_data=train_data,
                                                    test_data=test_data,
                                                    update_type="kfac",
                                                    task=args.task,
                                                    lr=lr,
                                                    epochs=args.epochs,
                                                    mini_batch_size=args.batch_size,
                                                    model=args.model_type,
                                                    )
            blockwise_ngd_memory_info = memory_usage(f, interval=1e-3)
            end_time = time()
            time_taken = end_time-start_time

            memory_time_info['block-ngd']['hidden-layer-num'].append(hidden_layer_num)
            memory_time_info['block-ngd']['time'].append(time_taken)
            memory_time_info['block-ngd']['avg-space'].append(np.mean(blockwise_ngd_memory_info))
            memory_time_info['block-ngd']['min-space'].append(np.min(blockwise_ngd_memory_info))
            memory_time_info['block-ngd']['max-space'].append(np.max(blockwise_ngd_memory_info))
            

        if args.use_exact_ngd:
            start_time = time()
            f = lambda : train(layers=[feature_num, *([args.hidden_layer_size]*hidden_layer_num), 1],
                                activations=['sigmoid' for i in range(hidden_layer_num+1)],
                                train_data=train_data,
                                test_data=test_data,
                                update_type="natural_gradient",
                                task=args.task,
                                lr=lr,
                                epochs=args.epochs,
                                mini_batch_size=args.batch_size,
                                model=args.model_type,
                                )
            exact_ngd_memory_info = memory_usage(f, interval=1e-3)
            end_time = time()
            time_taken = end_time-start_time

            memory_time_info['exact-ngd']['hidden-layer-num'].append(hidden_layer_num)
            memory_time_info['exact-ngd']['time'].append(time_taken)
            memory_time_info['exact-ngd']['avg-space'].append(np.mean(exact_ngd_memory_info))
            memory_time_info['exact-ngd']['min-space'].append(np.min(exact_ngd_memory_info))
            memory_time_info['exact-ngd']['max-space'].append(np.max(exact_ngd_memory_info))
            

    '''Plot info'''
    plt.figure(figsize=(10,5))

    fig_num = args.analyze_time + args.analyze_memory

    for method in ['tengrad', 'sgd', 'exact-ngd', 'block-ngd']:
        current_fig = 1
        if (method == 'tengrad' and args.use_tengrad) or (method == 'sgd' and args.use_sgd) or (method == 'block-ngd' and args.use_block_ngd) or (method == 'exact-ngd' and args.use_exact_ngd):
            if args.analyze_time:
                plt.subplot(fig_num,1,current_fig)
                plt.plot(memory_time_info[method]['hidden-layer-num'], (memory_time_info[method]['time']), label=method)
                plt.xlabel('hidden layer number')
                plt.ylabel("time(secs)")
                plt.legend(loc='upper right')
                plt.tight_layout()
                current_fig += 1
            if args.analyze_memory:
                plt.subplot(fig_num,1,current_fig)
                plt.plot(memory_time_info[method]['hidden-layer-num'], (memory_time_info[method]['max-space']), label=method)
                plt.xlabel('hidden layer number')
                plt.ylabel("max memory(MB)")
                plt.legend(loc='upper right')
                plt.tight_layout()
                current_fig += 1
    plt.show()
