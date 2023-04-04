import numpy as np
# import Wavelet_CNN__LSTM_Source_Network as Wavelet_CNN_Source_Network
import Wavelet_CNN_Source_Network as Wavelet_CNN_Source_Network
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import time
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
import torch.nn.functional as f
import load_evaluation_dataset
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


def calculate_fitness(examples_training, labels_training, examples_test_0, labels_test_0, examples_test_1,
                      labels_test_1):
    accuracy = []
    balanced_accuracy = []
    f1_macro = []
    precision_score = []
    recall_score = []

    # initialized_weights = np.load("initialized_weights.npy")
    for test_patient in range(17):
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

        print(torch.from_numpy(np.array(Y_fine_tune, dtype=np.int32)).size(0))
        print(np.shape(np.array(X_fine_tune, dtype=np.float32)))
        train = TensorDataset(torch.from_numpy(np.array(X_acc_train, dtype=np.float32)),
                              torch.from_numpy(np.array(Y_acc_train, dtype=np.int32)))
        validation = TensorDataset(torch.from_numpy(np.array(X_fine_tune, dtype=np.float32)),
                                   torch.from_numpy(np.array(Y_fine_tune, dtype=np.int32)))

        trainloader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
        validationloader = torch.utils.data.DataLoader(validation, batch_size=128, shuffle=True)

        test = TensorDataset(torch.from_numpy(np.array(X_test, dtype=np.float32)),
                               torch.from_numpy(np.array(Y_test, dtype=np.int32)))

        test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)

        cnn = Wavelet_CNN_Source_Network.Net(number_of_class=7, batch_size=128, number_of_channel=12,
                                             learning_rate=0.0404709, dropout=.5)

        criterion = nn.NLLLoss(size_average=False)
        optimizer = optim.Adam(cnn.parameters(), lr=0.0404709)

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
        precision_score.append(precision_score(all_ground_truth_labels, all_predicted_labels, average='macro'))
        recall_score.append(recall_score(all_ground_truth_labels, all_predicted_labels, average='macro'))
        cm = confusion_matrix(all_ground_truth_labels, all_predicted_labels, number_class=7)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        plt.figure()
        disp.plot()
        plt.savefig(f'CNN_Independent_Confusion_Matrix_{test_patient}')
        # conf_matrix_0.append(confusion_matrix(all_ground_truth_labels_0, all_predicted_labels_0))
        # report_0.append(classification_report(all_ground_truth_labels_0, all_predicted_labels_0))

        print("ACCURACY TEST FINAL : %.3f %%" % (100 * float(correct_prediction_test) / float(total)))
        # accuracy_test0.append(100 * float(correct_prediction_test_0) / float(total))

    print("AVERAGE ACCURACY TEST %.3f" % np.array(accuracy).mean())
    return accuracy, balanced_accuracy, f1_macro, precision_score, recall_score


def train_model(cnn, criterion, optimizer, scheduler, dataloaders, num_epochs=100, precision=1e-8):
    since = time.time()

    best_loss = float('inf')

    patience = 30
    patience_increase = 10
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
                    #print(outputs.shape, labels.long().shape)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    #loss = loss.data[0]

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
            print('{} Loss: {:.8f} Acc: {:.8}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss+precision < best_loss:
                    print("New best validation loss:", epoch_loss)
                    best_loss = epoch_loss
                    torch.save(cnn.state_dict(), 'best_weights_source_wavelet.pt')
                    patience = patience_increase + epoch
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - epoch_start))
        if epoch > patience:
            break
    print()

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    # load best model weights
    cnn_weights = torch.load('best_weights_source_wavelet.pt')
    # cnn_weights = torch.load('best_weights_source_wavelet.pt', map_location=torch.device('cpu'))
    cnn.load_state_dict(cnn_weights)
    cnn.eval()
    return cnn


if __name__ == '__main__':
    # Comment between here

    # examples, labels = load_evaluation_dataset.read_data('EvaluationDataset',
    #                                             type='training0')
    #
    # datasets = [examples, labels]
    # pickle.dump(datasets, open("saved_dataset_training.p", "wb"))
    #
    # examples, labels = load_evaluation_dataset.read_data('EvaluationDataset',
    #                                             type='Test0')
    #
    # datasets = [examples, labels]
    # pickle.dump(datasets, open("saved_dataset_test0.p", "wb"))
    #
    # examples, labels = load_evaluation_dataset.read_data('EvaluationDataset',
    #                                             type='Test1')
    #
    # datasets = [examples, labels]
    # pickle.dump(datasets, open("saved_dataset_test1.p", "wb"))

    # and here if the evaluation dataset was already processed and saved with "load_evaluation_dataset"
    import os

    print(os.listdir("../"))

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
    precision = []
    recall = []
    for i in range(3):
        print("ROUND: ", i)
        accuracy, balanced_accuracy, f1_macro, precision_score, recall_score = calculate_fitness(examples_training, labels_training,
                                                               examples_validation0, labels_validation0,
                                                               examples_validation1, labels_validation1)
        print(accuracy)

        acc.append(accuracy)
        bal.append(balanced_accuracy)
        f1.append(f1_macro)
        precision.append(precision_score)
        recall.append(recall_score)

        result_name = "cnn_source_independent_results.txt"

        with open(result_name, "w") as myfile:
            myfile.write("CNN STFT: \n\n")
            myfile.write("Accuracy: \n")
            myfile.write(str(np.mean(acc)) + '\n')
            myfile.write("Balanced Accuracy Score: \n")
            myfile.write(str(np.mean(bal)) + '\n')
            myfile.write("F1 Macro: \n")
            myfile.write(str(np.mean(f1)) + '\n')
            myfile.write("Precision: \n")
            myfile.write(str(np.mean(precision)) + '\n')
            myfile.write("Recall: \n")
            myfile.write(str(np.mean(recall)) + '\n\n')