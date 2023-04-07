import numpy as np
import Wavelet_CNN_LSTM_Target_Network as Wavelet_CNN_Target_Network
# import Wavelet_CNN_Target_Network as Wavelet_CNN_Target_Network
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
from scipy.stats import mode
import load_evaluation_dataset
import load_pre_training_dataset
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
import pickle

def confusion_matrix(pred, Y, number_class=7):
    confusion_matrice = []
    for x in range(0, number_class):
        vector = []
        for y in range(0, number_class):
            vector.append(0)
        confusion_matrice.append(vector)
    for prediction, real_value in zip(pred, Y):
        prediction = int(prediction)
        real_value = int(real_value)
        confusion_matrice[prediction][real_value] += 1
    return np.array(confusion_matrice)

def scramble(examples, labels, second_labels=[]):
    random_vec = np.arange(len(labels))
    np.random.shuffle(random_vec)
    new_labels = []
    new_examples = []
    if len(second_labels) == len(labels):
        new_second_labels = []
        for i in random_vec:
            new_labels.append(labels[i])
            new_examples.append(examples[i])
            new_second_labels.append(second_labels[i])
        return new_examples, new_labels, new_second_labels
    else:
        for i in random_vec:
            new_labels.append(labels[i])
            new_examples.append(examples[i])
        return new_examples, new_labels

def calculate_pre_training(examples, labels):
    list_train_dataloader = []
    list_validation_dataloader = []
    human_number = 0
    for j in range(19):
        examples_personne_training = []
        labels_gesture_personne_training = []
        labels_human_personne_training = []

        examples_personne_valid = []
        labels_gesture_personne_valid = []
        labels_human_personne_valid = []

        for k in range(len(examples[j])):
            if k < 21:
                examples_personne_training.extend(examples[j][k])
                labels_gesture_personne_training.extend(labels[j][k])
                labels_human_personne_training.extend(human_number * np.ones(len(labels[j][k])))
            else:
                examples_personne_valid.extend(examples[j][k])
                labels_gesture_personne_valid.extend(labels[j][k])
                labels_human_personne_valid.extend(human_number * np.ones(len(labels[j][k])))

        # print(np.shape(examples_personne_training))
        examples_personne_scrambled, labels_gesture_personne_scrambled, labels_human_personne_scrambled = scramble(
            examples_personne_training, labels_gesture_personne_training, labels_human_personne_training)

        examples_personne_scrambled_valid, labels_gesture_personne_scrambled_valid, labels_human_personne_scrambled_valid = scramble(
            examples_personne_valid, labels_gesture_personne_valid, labels_human_personne_valid)

        train = TensorDataset(torch.from_numpy(np.array(examples_personne_scrambled, dtype=np.float32)),
                              torch.from_numpy(np.array(labels_gesture_personne_scrambled, dtype=np.int32)))
        validation = TensorDataset(torch.from_numpy(np.array(examples_personne_scrambled_valid, dtype=np.float32)),
                                   torch.from_numpy(np.array(labels_gesture_personne_scrambled_valid, dtype=np.int32)))

        trainLoader = torch.utils.data.DataLoader(train, batch_size=3315, shuffle=True, drop_last=True)
        validationLoader = torch.utils.data.DataLoader(validation, batch_size=1312, shuffle=True, drop_last=True)

        list_train_dataloader.append(trainLoader)
        list_validation_dataloader.append(validationLoader)

        human_number += 1
        # print("Shape training : ", np.shape(examples_personne_scrambled))
        # print("Shape valid : ", np.shape(examples_personne_scrambled_valid))

    cnn = Wavelet_CNN_Target_Network.SourceNetwork(number_of_class=7, dropout_rate=.35)

    criterion = nn.NLLLoss(size_average=False)
    optimizer = optim.Adam(cnn.parameters(), lr=0.0404709)
    precision = 1e-8
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=15,
                                                     verbose=True, eps=precision)

    pre_train_model(cnn, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                    dataloaders={"train": list_train_dataloader, "val": list_validation_dataloader},
                    precision=precision)

def pre_train_model(cnn, criterion, optimizer, scheduler, dataloaders, num_epochs=10, precision=1e-8):
    since = time.time()

    # Create a list of dictionaries that will hold the weights of the batch normalisation layers for each dataset
    #  (i.e. each participants)
    list_dictionaries_BN_weights = []
    for index_BN_weights in range(len(dataloaders['val'])):
        state_dict = cnn.state_dict()
        batch_norm_dict = {}
        for key in state_dict:
            if "batch_norm" in key:
                batch_norm_dict.update({key: state_dict[key]})
        list_dictionaries_BN_weights.append(copy.deepcopy(batch_norm_dict))

    best_loss = float('inf')

    best_weights = copy.deepcopy(cnn.state_dict())

    patience = 30
    patience_increase = 30
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                cnn.train(True)  # Set model to training mode
            else:
                cnn.train(False)  # Set model to evaluate mode

            running_loss = 0.
            running_corrects = 0
            total = 0

            # Get a random order for the training dataset
            random_vec = np.arange(len(dataloaders[phase]))
            np.random.shuffle(random_vec)

            for dataset_index in random_vec:
                # Retrieves the BN weights calculated so far for this dataset
                BN_weights = list_dictionaries_BN_weights[dataset_index]
                cnn.load_state_dict(BN_weights, strict=False)

                loss_over_datasets = 0.
                correct_over_datasets = 0.
                for i, data in enumerate(dataloaders[phase][dataset_index], 0):
                    # get the inputs
                    inputs, labels = data

                    inputs, labels = Variable(inputs), Variable(labels)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    if phase == 'train':
                        cnn.train()
                        # forward
                        outputs = cnn(inputs)
                        _, predictions = torch.max(outputs.data, 1)

                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()
                        # print(loss.data)
                        # loss = loss.data[0]

                    else:
                        cnn.eval()

                        accumulated_predicted = Variable(torch.zeros(len(inputs), 7))
                        loss_intermediary = 0.
                        total_sub_pass = 0
                        for repeat in range(20):
                            outputs = cnn(inputs)
                            loss = criterion(outputs, labels.long())
                            if loss_intermediary == 0.:
                                loss_intermediary = loss
                            else:
                                loss_intermediary += loss
                            _, prediction_from_this_sub_network = torch.max(outputs.data, 1)
                            accumulated_predicted[range(len(inputs)),
                                                  prediction_from_this_sub_network.cpu().numpy().tolist()] += 1
                            total_sub_pass += 1
                        _, predictions = torch.max(accumulated_predicted.data, 1)
                        loss = loss_intermediary / total_sub_pass
                    # Statistic for this dataset
                    loss_over_datasets += loss
                    correct_over_datasets += torch.sum(predictions == labels.data)
                    total += labels.size(0)
                # Statistic global
                running_loss += loss_over_datasets
                running_corrects += correct_over_datasets

                # Save the BN statistics for this dataset
                state_dict = cnn.state_dict()
                batch_norm_dict = {}
                for key in state_dict:
                    if "batch_norm" in key:
                        batch_norm_dict.update({key: state_dict[key]})
                list_dictionaries_BN_weights[dataset_index] = copy.deepcopy(batch_norm_dict)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            # print('{} Loss: {:.8f} Acc: {:.8}'.format(
            #     phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss + precision < best_loss:
                    # print("New best validation loss:", epoch_loss)
                    best_loss = epoch_loss
                    best_weights = copy.deepcopy(cnn.state_dict())
                    patience = patience_increase + epoch
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - epoch_start))
        if epoch > patience:
            break

    # print()

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # Save the best weights found to file
    torch.save(best_weights, 'best_pre_train_weights_target_wavelet.pt')

def calculate_fitness(examples_training, labels_training, examples_test_0, labels_test_0, examples_test_1,
                      labels_test_1):
    accuracy = []
    balanced_accuracy = []
    f1_macro = []
    prec_score = []
    rec_score = []

    # initialized_weights = np.load("initialized_weights.npy")
    for test_patient in range(17):
        print(test_patient)
        X_train = []
        Y_train = []
        X_test, Y_test = [], []

        for j in range(17):
            for k in range(len(examples_training[j])):
                if j == test_patient:
                    X_test.extend(examples_training[j][k])
                    Y_test.extend(labels_training[j][k])
                    X_test.extend(examples_test_0[j][k])
                    Y_test.extend(labels_test_0[j][k])
                    X_test.extend(examples_test_1[j][k])
                    Y_test.extend(labels_test_1[j][k])
                else:
                    if k < 28:
                        X_train.extend(examples_training[j][k])
                        Y_train.extend(labels_training[j][k])

        X_train, Y_train = scramble(X_train, Y_train)
        X_test, Y_test = scramble(X_test, Y_test)

        X_train = X_train[0:int(len(X_train) * 0.2)]
        Y_train = Y_train[0:int(len(Y_train) * 0.2)]

        X_acc_train = X_train[0:int(len(X_train) * 0.9)]
        Y_acc_train = Y_train[0:int(len(Y_train) * 0.9)]

        X_fine_tune = X_train[int(len(X_train) * 0.9):]
        Y_fine_tune = Y_train[int(len(Y_train) * 0.9):]

        X_test = X_test[0:int(len(X_test) * 0.2)]
        Y_test = Y_test[0:int(len(Y_test) * 0.2)]

        # print(torch.from_numpy(np.array(Y_fine_tune, dtype=np.int32)).size(0))
        # print(np.shape(np.array(X_fine_tune, dtype=np.float32)))
        train = TensorDataset(torch.from_numpy(np.array(X_acc_train, dtype=np.float32)),
                              torch.from_numpy(np.array(Y_acc_train, dtype=np.int32)))
        validation = TensorDataset(torch.from_numpy(np.array(X_fine_tune, dtype=np.float32)),
                                   torch.from_numpy(np.array(Y_fine_tune, dtype=np.int32)))

        trainloader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
        validationloader = torch.utils.data.DataLoader(validation, batch_size=128, shuffle=True)

        test = TensorDataset(torch.from_numpy(np.array(X_test, dtype=np.float32)),
                             torch.from_numpy(np.array(Y_test, dtype=np.int32)))

        test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)

        pre_trained_weights = torch.load('best_pre_train_weights_target_wavelet.pt', map_location=torch.device('cpu'))

        cnn = Wavelet_CNN_Target_Network.TargetNetwork(number_of_class=7,
                                                       weights_pre_trained_cnn=pre_trained_weights)

        criterion = nn.NLLLoss(size_average=False)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, cnn.parameters()), lr=0.0404709)

        precision = 1e-8
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                         verbose=True, eps=precision)

        cnn = train_model(cnn, criterion, optimizer, scheduler,
                          dataloaders={"train": trainloader, "val": validationloader}, precision=precision)

        cnn.eval()
        total = 0
        correct_prediction_test = 0
        # Create empty arrays to store predicted and ground truth labels
        all_predicted_labels = []
        all_ground_truth_labels = []
        for k, data_test in enumerate(test_loader, 0):
            # get the inputs
            inputs_test, ground_truth_test = data_test
            inputs_test, ground_truth_test = Variable(inputs_test), Variable(ground_truth_test)

            concat_input = inputs_test
            for i in range(20):
                concat_input = torch.cat([concat_input, inputs_test])
            outputs_test = cnn(concat_input)
            _, predicted = torch.max(outputs_test.data, 1)
            correct_prediction_test += (mode(predicted.cpu().numpy())[0][0] ==
                                          ground_truth_test.data.cpu().numpy()).sum()

            # Append predicted and ground truth labels to the arrays
            all_predicted_labels.append(mode(predicted.cpu().numpy())[0][0])
            all_ground_truth_labels.append(ground_truth_test.data.cpu().numpy())

            total += ground_truth_test.size(0)

        # Convert the arrays to NumPy arrays
        all_predicted_labels = np.array(all_predicted_labels)
        all_ground_truth_labels = np.concatenate(all_ground_truth_labels)

        # Calculate the metrics
        accuracy.append(accuracy_score(all_ground_truth_labels, all_predicted_labels))
        balanced_accuracy.append(balanced_accuracy_score(all_ground_truth_labels, all_predicted_labels))
        f1_macro.append(f1_score(all_ground_truth_labels, all_predicted_labels, average='macro'))
        prec_score.append(precision_score(all_ground_truth_labels, all_predicted_labels, average='macro'))
        rec_score.append(recall_score(all_ground_truth_labels, all_predicted_labels, average='macro'))
        cm = confusion_matrix(all_ground_truth_labels, all_predicted_labels, number_class=7)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        plt.figure()
        disp.plot()
        plt.savefig(f'CNN_LSTM_Independent_TF_Confusion_Matrix_{test_patient}')

        print("ACCURACY TEST FINAL: %.3f %%" % (100 * float(correct_prediction_test) / float(total)))

    print("AVERAGE ACCURACY %.3f" % np.array(accuracy).mean())
    return accuracy, balanced_accuracy, f1_macro, prec_score, rec_score

def train_model(cnn, criterion, optimizer, scheduler, dataloaders, num_epochs=75, precision=1e-8):
    since = time.time()

    best_loss = float('inf')

    patience = 30
    patience_increase = 10

    best_weights = copy.deepcopy(cnn.state_dict())

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                cnn.train(True)  # Set model to training mode
            else:
                cnn.train(False)  # Set model to evaluate mode

            running_loss = 0.
            running_corrects = 0
            total = 0

            for i, data in enumerate(dataloaders[phase], 0):
                # get the inputs
                inputs, labels = data

                inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                if phase == 'train':
                    cnn.train()
                    # forward
                    outputs = cnn(inputs)
                    _, predictions = torch.max(outputs.data, 1)

                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()
                    # loss = loss.data[0]

                else:
                    cnn.eval()

                    accumulated_predicted = Variable(torch.zeros(len(inputs), 7))
                    loss_intermediary = 0.
                    total_sub_pass = 0
                    for repeat in range(20):
                        outputs = cnn(inputs)
                        loss = criterion(outputs, labels.long())
                        if loss_intermediary == 0.:
                            loss_intermediary = loss
                        else:
                            loss_intermediary += loss
                        _, prediction_from_this_sub_network = torch.max(outputs.data, 1)
                        accumulated_predicted[range(len(inputs)),
                                              prediction_from_this_sub_network.cpu().numpy().tolist()] += 1
                        total_sub_pass += 1
                    _, predictions = torch.max(accumulated_predicted.data, 1)
                    loss = loss_intermediary/total_sub_pass



                # statistics
                running_loss += loss
                running_corrects += torch.sum(predictions == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            # print('{} Loss: {:.8f} Acc: {:.8}'.format(
            #     phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss+precision < best_loss:
                    # print("New best validation loss:", epoch_loss)
                    best_loss = epoch_loss
                    best_weights = copy.deepcopy(cnn.state_dict())
                    patience = patience_increase + epoch
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - epoch_start))
        if epoch > patience:
            break
    # print()

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val loss: {:4f}'.format(best_loss))
    # Save to file the best weights found
    torch.save(best_weights, 'best_weights_source_wavelet_target.pt')
    # load best model weights
    cnn.load_state_dict(copy.deepcopy(best_weights))
    cnn.eval()
    return cnn


if __name__ == '__main__':

    # examples, labels = load_evaluation_dataset.read_data('EvaluationDataset',
    #                                             type='training0')
    #
    # datasets = [examples, labels]
    # np.save("saved_dataset_training.p", datasets)
    #
    # examples, labels = load_evaluation_dataset.read_data('EvaluationDataset',
    #                                             type='Validation0')
    #
    # datasets = [examples, labels]
    # np.save("saved_dataset_test0.p", datasets)
    #
    # examples, labels = load_evaluation_dataset.read_data('EvaluationDataset',
    #                                             type='Validation1')
    #
    # datasets = [examples, labels]
    # np.save("saved_dataset_test1.p", datasets)


    # Comment between here

    # examples, labels = load_pre_training_dataset.read_data('PreTrainingDataset')
    # datasets = [examples, labels]
    #
    # pickle.dump(datasets, open("saved_pre_training_dataset_pickle.p", "wb"))
    #
    # np.save("saved_pre_training_dataset.p", datasets)

    # And here if the pre-training dataset was already processed and saved

    # Comment between here

    datasets_pre_training = np.load("saved_pre_training_dataset.p", encoding="bytes", allow_pickle=True)
    examples_pre_training, labels_pre_training = datasets_pre_training

    calculate_pre_training(examples_pre_training, labels_pre_training)

    # And here if the pre-training of the network was already completed.

    datasets_training = np.load("saved_dataset_training.p", encoding="bytes", allow_pickle=True)
    examples_training, labels_training = datasets_training

    datasets_validation0 = np.load("saved_dataset_test0.p", encoding="bytes", allow_pickle=True)
    examples_validation0, labels_validation0 = datasets_validation0

    datasets_validation1 = np.load("saved_dataset_test1.p", encoding="bytes", allow_pickle=True)
    examples_validation1, labels_validation1 = datasets_validation1
    # print("SHAPE", np.shape(examples_training))

    accuracy_one_by_one = []
    array_training_error = []
    array_validation_error = []

    acc = []
    bal = []
    f1 = []
    prec = []
    rec = []

    for i in range(3):
        print("ROUND: ", i)
        accuracy, balanced_accuracy, f1_macro, prec_score, rec_score = calculate_fitness(examples_training,
                                                                                                 labels_training,
                                                                                                 examples_validation0,
                                                                                                 labels_validation0,
                                                                                                 examples_validation1,
                                                                                                 labels_validation1)
        print(accuracy)

        acc.append(accuracy)
        bal.append(balanced_accuracy)
        f1.append(f1_macro)
        prec.append(prec_score)
        rec.append(rec_score)

        result_name = "cnn_lstm_target_independent_results.txt"

        with open(result_name, "w") as myfile:
            myfile.write("CNN STFT: \n\n")
            myfile.write("Accuracy: \n")
            myfile.write(str(np.mean(acc)) + '\n')
            myfile.write("Balanced Accuracy Score: \n")
            myfile.write(str(np.mean(bal)) + '\n')
            myfile.write("F1 Macro: \n")
            myfile.write(str(np.mean(f1)) + '\n')
            myfile.write("Precision: \n")
            myfile.write(str(np.mean(prec)) + '\n')
            myfile.write("Recall: \n")
            myfile.write(str(np.mean(rec)) + '\n\n')