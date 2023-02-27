from .models.DNN import DNN
from .losses.BCE import BCE
from .losses.MSE import MSE
from tqdm import tqdm
import numpy as np


def train(layers,
          activations,
          train_data,
          test_data,
          update_type="first_order",
          task='classification',
          lr=1e-4,
          epochs=10000,
          mini_batch_size=32,
          lambda_reg=0,
          model='dnn',
          lr_decay=False,
          lr_decay_type='smooth',
          running_loss_window=1,
          alpha=1e-6):
    '''Train model based on passed arguments'''
    if model == 'dnn':
        if update_type == 'first_order':
            model = DNN(layers, activations,
                        lambda_reg=lambda_reg, is_ngd=False)
        else:
            model = DNN(layers, activations,
                        lambda_reg=lambda_reg)

    if task == 'classification':
        loss = BCE()
        train_info = {'iter_wise': {'iterations': [], 'loss': [], 'running_loss': []}, 'epoch_wise': {
            'epochs': [], 'loss': [], 'val_loss': [], 'val_accuracy': []}}
        validation_accuracy = 0

    elif task == 'regression':
        loss = MSE()
        train_info = {'iter_wise': {'iterations': [], 'loss': [], 'running_loss': []},
                      'epoch_wise': {'epochs': [], 'loss': [], 'val_loss': []}}
    else:
        raise Exception(f"Unknown task type: {task} passed!!!")

    pbar = tqdm(list(range(1, epochs+1)))

    train_features, train_labels = train_data[:-1], train_data[-1:]
    test_features, test_labels = test_data[:-1], test_data[-1:]

    feature_num, train_num, test_num = train_features.shape[
        0], train_features.shape[1], test_features.shape[1]

    iteration = 0
    average_loss = float("inf")
    val_loss = float("inf")

    '''Log Initial epoch wise train info'''
    initial_train_pred, initial_test_pred = model.forward(
        train_features), model.forward(test_features)
    train_info['epoch_wise']['epochs'].append(0)
    train_info['epoch_wise']['loss'].append(
        loss.compute(initial_train_pred, train_labels))
    train_info['epoch_wise']['val_loss'].append(
        loss.compute(initial_test_pred, test_labels))

    if task == 'classification':
        train_info['epoch_wise']['val_accuracy'].append(
            np.mean(np.where(initial_test_pred > 0.5, 1, 0) == test_labels))

    lr_ = lr
    try:
        for epoch in pbar:
            data_idxs = set(range(train_num))
            epoch_loss_lst = []
            while data_idxs:
                iteration += 1

                '''Decaying of lr'''
                if lr_decay and lr_decay_type == 'smooth':
                    lr_ = lr/(1+0.1*iteration//10)

                '''Sample batch'''
                sample_idx = np.random.choice(list(data_idxs), size=min(
                    mini_batch_size, len(data_idxs)), replace=False).tolist()
                data_idxs -= set(sample_idx)

                '''Forward pass, backward pass and update model'''
                data, labels = \
                    train_features[:, sample_idx], train_labels[:, sample_idx]
                out = model.forward(data)
                loss_val = loss.compute(out, labels)
                epoch_loss_lst.append(loss_val)
                loss_grad = loss.backward()
                model.backward(loss_grad)
                model.update(lr_, update_type, alpha=alpha)

                '''Break if loss exploding'''
                if np.isnan(loss_val) or np.log(loss_val) > 20:
                    return train_info

                '''Log iter wise train info'''
                train_info['iter_wise']['iterations'].append(iteration)
                train_info['iter_wise']['loss'].append(loss_val)
                train_info['iter_wise']['running_loss'].append(
                    np.mean(train_info['iter_wise']['loss'][-running_loss_window:]))

                '''Show training progress'''
                if task == "classification":
                    pbar.set_description("Update Type: %s | LR: %f Iteration: %d Epoch: %d Loss: %f "
                                        "Epoch Loss: %f Val Loss: %f Validation Accuracy %f" % (update_type,
                                                                                                lr_,
                                                                                                iteration,
                                                                                                epoch,
                                                                                                round(
                                                                                                    loss_val, 6),
                                                                                                round(
                                                                                                    average_loss, 6),
                                                                                                round(
                                                                                                    val_loss, 6),
                                                                                                round(validation_accuracy, 6)))

                elif task == "regression":
                    pbar.set_description("Update Type: %s | LR: %f Iteration: %d Epoch: %d Loss: %f "
                                        "Epoch Loss: %f Val Loss: %f" % (update_type,
                                                                        lr_,
                                                                        iteration,
                                                                        epoch,
                                                                        round(
                                                                            loss_val, 6),
                                                                        round(
                                                                            average_loss, 6),
                                                                        round(val_loss, 6)))

            '''Decaying of lr'''
            if lr_decay and lr_decay_type == 'step':
                lr_ = lr_/2

            '''Log averate loss in training data and metrics in test data'''
            average_loss = np.mean(epoch_loss_lst)
            if task == 'classification':
                pred_prob = model.forward(test_features)
                preds = np.where(pred_prob > 0.5, 1, 0)
                val_loss = loss.compute(pred_prob, test_labels)
                validation_accuracy = np.mean(test_labels == preds)
                train_info['epoch_wise']['val_accuracy'].append(
                    validation_accuracy)
            else:
                pred = model.forward(test_features)
                val_loss = loss.compute(pred, test_labels)

            '''Log epoch wise train info'''
            train_info['epoch_wise']['epochs'].append(epoch)
            train_info['epoch_wise']['loss'].append(average_loss)
            train_info['epoch_wise']['val_loss'].append(val_loss)
    except Exception as e:
        print(e)
    # print(train_info['epoch_wise'])
    return train_info
