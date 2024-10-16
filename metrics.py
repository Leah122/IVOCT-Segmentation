import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif

# inspired by https://gitlab.com/python-packages2/monte-carlo-analysis/-/blob/pypi/monte_carlo_analysis/metrics/


def AUCPR(prediction, uncertainty_map, ground_truth):
    class_prediction = np.argmax(prediction.mean(axis=0), axis=0)
    class_ground_truth = ground_truth.squeeze()

    misclassification_map = (class_ground_truth != class_prediction)
    return average_precision_score(misclassification_map.ravel(), uncertainty_map.ravel())


def AUCPRClassWise(prediction, uncertainty_map, ground_truth):
    nb_classes = prediction.shape[1]

    class_prediction = np.argmax(prediction.mean(axis=0), axis=0)
    class_ground_truth = np.argmax(ground_truth, axis=0)

    results = []
    for i in range(nb_classes):
        class_ground_truth_ = (class_ground_truth == i)
        class_prediction_ = (class_prediction == i)
        misclassification_map = (class_ground_truth_ != class_prediction_)
        results.append(
                average_precision_score(
                    misclassification_map.ravel(), uncertainty_map[i].ravel()
                    )
                )
    return results


def AUCROC(prediction, uncertainty_map, ground_truth):
    class_prediction = np.argmax(prediction.mean(axis=0), axis=0)
    class_ground_truth = ground_truth.squeeze()

    # Compute misclassification map
    misclassification_map = (class_ground_truth != class_prediction)
    print(misclassification_map.shape)

    return roc_auc_score(
            misclassification_map.ravel(), uncertainty_map.ravel()
            )

# def numeric_score(predictions, labels, threshold):
#     """
#     predictions of shape (nr_mc_samples, 1, img, img)
#     labels of shape (1, img, img)
#     returns FP, FN, TP, TN """

#     FP = 

#     # FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
#     # FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
#     # TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
#     # TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))

#     return FP, FN, TP, TN


def entropy(outputs):
    mean_prediction_per_class = np.mean(outputs, axis=0)
    
    # set 0's to 1 to prevent log of 0
    zeros = mean_prediction_per_class == 0
    mean_prediction_per_class[zeros] = 1

    return -np.sum(mean_prediction_per_class * np.log(mean_prediction_per_class), axis=0)

def expectation(outputs):
    # set 0's to 1 to prevent log of 0
    zeros = outputs == 0
    outputs[zeros] = 1

    log_prediction = np.log(outputs)
    return -1.0/outputs.shape[0] * np.sum(np.sum(log_prediction * outputs, axis=1), axis=0)


def MI(predictions):
    '''
        predictions of shape (nr_mc_samples, nr_classes, image_dim, image_dim)
        labels of shape (1, image_dim, image_dim)
        main source: Exploring uncertainty measures in deep networks for Multiple sclerosis lesion detection and segmentation.pdf
    '''

    entropies = entropy(predictions)
    expectations = expectation(predictions)
    
    return entropies - expectations

    # print("MI")
    # print(predictions.shape)

    # mi = np.zeros(labels.shape)


    # for i in range(labels.shape[1]):
    #     for j in range(labels.shape[2]):
    #         # one hot label
    #         one_hot_label = np.zeros(15)
    #         one_hot_label[labels[0,i,j]] = 1

    #         # probabilities per class
    #         predictions_per_class = np.zeros((15))
    #         for pred in predictions[:, i, j]:
    #             predictions_per_class[pred] += 1
            
    #         predicted_probabilities = predictions_per_class / predictions_per_class.sum()

    #         predicted_probabilities = predicted_probabilities[np.where(predicted_probabilities) == 0] = 1

    #         print("one_hot_label ", one_hot_label)
    #         print("predictions: ", predictions[:, i, j])
    #         print("predictions: ", predicted_probabilities)
    #         mi = one_hot_label * np.log(predicted_probabilities)
    #         print(mi)
            # np.sum(one_hot_label, predictions[:, i, j])

    # turn data into nr of predictions per class
    # predictions_argmaxed = np.argmax(predictions, axis=1)
    # predictions_transformed = np.zeros((15, predictions_argmaxed.shape[1], predictions_argmaxed.shape[2]))
    # for i in range(predictions_argmaxed.shape[1]):
    #     for j in range(predictions_argmaxed.shape[2]):
    #         predictions_per_class = np.zeros((15))
    #         for pred in predictions_argmaxed[:, i, j]:
    #             predictions_per_class[pred] += 1
    #         predictions_transformed[:, i, j] = predictions_per_class
    
    # print(predictions_transformed.shape)
    # print(labels.detach().cpu().numpy().squeeze().shape)



    # entropies = joint_entropies.diagonal()
    # entropies_tile = np.tile(entropies, (704, 704, 1))
    # sum_entropies = entropies_tile + entropies_tile.T
    # mi_matrix = sum_entropies - joint_entropies
    # norm_mi_matrix = mi_matrix * 2 / sum_entropies
    # print(norm_mi_matrix.shape)


    # print(en.shape)

    # mi_matrix = mutual_info_classif(predictions_transformed.reshape(15, -1).swapaxes(0,1), labels.reshape(-1), discrete_features=True)
    
    # mi_matrix.reshape(704,704)

    # print(mi_matrix)

    # −(0⋅log(0.2)+1⋅log(0.6)+0⋅log(0.2))