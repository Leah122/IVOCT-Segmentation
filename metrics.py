import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from constants import NEW_CLASSES, NEW_CLASS_DICT
import torch
from utils import soft_dice, dice_score

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

    return roc_auc_score(
            misclassification_map.ravel(), uncertainty_map.ravel()
            )

def AUCROCClassWise(prediction, uncertainty_map, ground_truth):
    class_prediction = np.argmax(prediction.mean(axis=0), axis=0)
    class_ground_truth = ground_truth.squeeze()

    results = []
    for i in range(NEW_CLASSES):
        class_ground_truth_ = (class_ground_truth == i)
        class_prediction_ = (class_prediction == i)
        misclassification_map = (class_ground_truth_ != class_prediction_)
        if len(set(misclassification_map.ravel())) > 1:
            results.append(
                    roc_auc_score(
                        misclassification_map.ravel(), uncertainty_map[i].ravel()
                        )
                    )
        else: 
            results.append(None)
    return results

def ROC_curve(prediction, uncertainty_map, ground_truth):
    class_prediction = np.argmax(prediction.mean(axis=0), axis=0)
    class_ground_truth = ground_truth.squeeze()

    # Compute misclassification map
    misclassification_map = (class_ground_truth != class_prediction)

    fpr, tpr, thresholds = roc_curve(misclassification_map.ravel(), uncertainty_map.ravel())
    return fpr, tpr, thresholds


def entropy(outputs):
    mean_prediction_per_class = np.mean(outputs, axis=0)
    log_1 = np.log2(mean_prediction_per_class, out=np.zeros_like(mean_prediction_per_class, dtype=np.float64), where=(mean_prediction_per_class!=0))
    return (-np.sum(mean_prediction_per_class * log_1, axis=0)) / np.log2(NEW_CLASSES)


# def entropy_per_class(outputs): 
#     mean_prediction_per_class = np.mean(outputs, axis=0)
#     log_1 = np.log2(mean_prediction_per_class, out=np.zeros_like(mean_prediction_per_class, dtype=np.float64), where=(mean_prediction_per_class!=0))
#     log_2 = np.log2(1-mean_prediction_per_class, out=np.zeros_like(mean_prediction_per_class, dtype=np.float64), where=(1-mean_prediction_per_class!=0))
#     return (-mean_prediction_per_class * log_1) - ((1-mean_prediction_per_class) * log_2) / np.log2(2)


def entropy_per_class(outputs): 
    log_1 = np.log2(outputs, out=np.zeros_like(outputs, dtype=np.float64), where=(outputs!=0))
    entropy = (-np.sum(outputs * log_1, axis=0)) / np.log(outputs.shape[0])
    return entropy


def entropy_per_class_avg(outputs): 
    uncertainty_map = np.zeros((NEW_CLASSES, 704, 704))
    mean_prediction_per_class = np.mean(outputs, axis=0)

    for clas in range(NEW_CLASSES):
        classes = NEW_CLASSES * [True]
        classes[clas] = False

        clas1 = mean_prediction_per_class[clas]
        clas2 = np.mean(mean_prediction_per_class[classes], axis=0)
        total = clas1 + clas2

        clas1 = clas1 / total
        clas2 = clas2 / total

        log_1 = np.log2(clas1, out=np.zeros_like(clas1, dtype=np.float64), where=(clas1!=0))
        log_2 = np.log2(clas2, out=np.zeros_like(clas2, dtype=np.float64), where=(clas2!=0))
        uncertainty_map[clas] = (-clas1 * log_1) - (clas2 * log_2)
    
    return uncertainty_map


def entropy_per_class_test(outputs): 
    uncertainty_map = np.zeros((NEW_CLASSES, 1, 1))
    mean_prediction_per_class = np.mean(outputs, axis=0)
    
    for clas in range(NEW_CLASSES):
        classes = NEW_CLASSES * [True]
        classes[clas] = False

        clas1 = mean_prediction_per_class[clas]
        clas2 = mean_prediction_per_class[classes]

        log_1 = np.log2(clas1, out=np.zeros_like(clas1, dtype=np.float64), where=(clas1!=0))
        log_2 = np.log2(clas2, out=np.zeros_like(clas2, dtype=np.float64), where=(clas2!=0))
        print((-(clas1 * log_1)), np.sum(-(clas2 * log_2), axis=0)/9)
        uncertainty_map[clas] = ((-(clas1 * log_1) + np.sum(-(clas2 * log_2), axis=0)/9))/2 * 10 / np.log2(NEW_CLASSES)
    
    return uncertainty_map


def expectation(outputs):
    log_prediction = np.log2(outputs, out=np.zeros_like(outputs, dtype=np.float64), where=(outputs!=0))
    return (-1.0/outputs.shape[0] * np.sum(np.sum(log_prediction * outputs, axis=1), axis=0)) / np.log2(NEW_CLASSES)


def expectation_per_class(outputs):
    log_prediction = np.log2(outputs, out=np.zeros_like(outputs, dtype=np.float64), where=(outputs!=0))
    return (-1.0/outputs.shape[1] * np.sum(log_prediction * outputs, axis=0))# / np.log(outputs.shape[0])

# def expectation_per_class(outputs):
#     log_1 = np.log2(outputs, out=np.zeros_like(outputs, dtype=np.float64), where=(outputs!=0))
#     log_2 = np.log2(1-outputs, out=np.zeros_like(outputs, dtype=np.float64), where=(1-outputs!=0))
#     return (-1.0/outputs.shape[0] * np.sum((log_1 * outputs) + (log_2 * (1 - outputs)), axis=0)) / np.log2(2)


def expectation_per_class_avg(outputs):
    uncertainty_map = np.zeros((NEW_CLASSES, 704, 704))

    for clas in range(NEW_CLASSES):
        classes = NEW_CLASSES * [True]
        classes[clas] = False

        # print(outputs.shape)

        clas1 = outputs[:, clas]
        clas2 = np.mean(outputs[:, classes], axis=1)
        # print(clas1.shape, clas2.shape)
        total = clas1 + clas2

        clas1 = clas1 / total
        clas2 = clas2 / total

        # print(clas1[:,0,0])
        # print(clas2[:,0,0])

        # print(clas1.max(), clas1.min())
        # print(clas2.max(), clas2.min())

        log_1 = np.log2(clas1, out=np.zeros_like(clas1, dtype=np.float64), where=(clas1!=0))
        log_2 = np.log2(clas2, out=np.zeros_like(clas2, dtype=np.float64), where=(clas2!=0))

        # print(log_1.max(), log_1.min(), log_1.shape)
        # print(log_2.max(), log_2.min(), log_2.shape)
        # print((log_1[:,0,0] * clas1[:,0,0]) + (log_2[:,0,0] * clas2[:,0,0]))
    
        uncertainty_map[clas] = (-1.0/outputs.shape[0] * np.sum((log_1 * clas1) + (log_2 * clas2), axis=0))
        # uncertainty_map[clas] = (-clas1 * log_1) - (clas2 * log_2)
    
    return uncertainty_map

def MI(predictions):
    entropies = entropy(predictions)
    expectations = expectation(predictions)
    return entropies - expectations


def MI_per_class(predictions):
    entropies = entropy_per_class(predictions)
    expectations = expectation_per_class(predictions)
    return entropies - expectations


def MI_per_class_avg(predictions):
    entropies = entropy_per_class_avg(predictions)
    expectations = expectation_per_class_avg(predictions)
    return entropies - expectations
    

def brier_score(predictions):
    predictions_probabilities = predictions.mean(axis=0)
    predictions_one_hot = np.eye(NEW_CLASSES)[predictions_probabilities.argmax(axis=0)].swapaxes(1,2).swapaxes(0,1)
    mse = np.square(predictions_probabilities - predictions_one_hot)
    brier = mse.sum(axis=0) 
    return brier


def brier_score_per_class(predictions):
    predictions_probabilities = predictions.mean(axis=0)
    predictions_one_hot = np.eye(NEW_CLASSES)[predictions_probabilities.argmax(axis=0)].swapaxes(1,2).swapaxes(0,1)
    brier_per_class = np.square(predictions_probabilities - predictions_one_hot)
    return brier_per_class


def group_analysis(prediction, uncertainty_map, ground_truth, percentile=90, p=None):
    if p == None:
        p = np.percentile(uncertainty_map.flatten(), percentile)

    mean_predictions = prediction.mean(axis=0).argmax(axis=0)

    certain = np.where(uncertainty_map < p, uncertainty_map, -1)

    certain_group_predictions = np.where(certain > -1, mean_predictions, -1)
    certain_group_labels = np.where(certain > -1, ground_truth, -1)

    uncertain_group_predictions = np.where(certain == -1, mean_predictions, -1)
    uncertain_group_labels = np.where(certain == -1, ground_truth, -1)


    dice_per_class_certain = dice_score(certain_group_predictions, certain_group_labels.squeeze())
    # dice_certain = np.nanmean(dice_per_class_certain[1:])

    dice_per_class_uncertain = dice_score(uncertain_group_predictions, uncertain_group_labels.squeeze())
    # dice_uncertain = np.nanmean(dice_per_class_uncertain[1:])

    return dice_per_class_certain, dice_per_class_uncertain


def filter_uncertain_images(image_uncertainties, percentile=90):
    p = np.percentile(image_uncertainties, percentile)
    uncertain = (image_uncertainties >= p)
    print("threshold uncertain images: ", p)
    return uncertain


def sensitivity_specificity(outputs, labels):

    classes = [4, 5, 8, 9] # lipid, calcium, sidebranch, healed plaque
    predictions = outputs.mean(axis=0).argmax(axis=0)

    metrics = {}

    for clas in classes:
        TP = np.any(predictions == clas) and np.any(labels == clas)
        FP = np.any(predictions == clas) and (not np.any(labels == clas))
        TN = (not np.any(predictions == clas)) and (not np.any(labels == clas))
        FN = (not np.any(predictions == clas)) and np.any(labels == clas)

        metrics[clas] = np.array([int(TP), int(FP), int(TN), int(FN)])
    
    return metrics

