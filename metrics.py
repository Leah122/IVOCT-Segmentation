import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from constants import NEW_CLASSES, NEW_CLASS_DICT
import torch
from utils import soft_dice

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


# def variance(outputs):
#     variance = outputs.var(axis=0) #TODO: should this be 0 or 1?
#     return variance

def entropy(outputs):
    mean_prediction_per_class = np.mean(outputs, axis=0)
    log_1 = np.log2(mean_prediction_per_class, out=np.zeros_like(mean_prediction_per_class, dtype=np.float64), where=(mean_prediction_per_class!=0))
    return -np.sum(mean_prediction_per_class * log_1, axis=0)


def entropy_per_class(outputs):
    mean_prediction_per_class = np.mean(outputs, axis=0)
    log_1 = np.log2(mean_prediction_per_class, out=np.zeros_like(mean_prediction_per_class, dtype=np.float64), where=(mean_prediction_per_class!=0))
    log_2 = np.log2(1-mean_prediction_per_class, out=np.zeros_like(mean_prediction_per_class, dtype=np.float64), where=(1-mean_prediction_per_class!=0))
    return (-mean_prediction_per_class * log_1) - ((1-mean_prediction_per_class) * log_2)


def expectation(outputs):
    log_prediction = np.log2(outputs, out=np.zeros_like(outputs, dtype=np.float64), where=(outputs!=0))
    return -1.0/outputs.shape[0] * np.sum(np.sum(log_prediction * outputs, axis=1), axis=0)


def expectation_per_class(outputs):
    log_1 = np.log2(outputs, out=np.zeros_like(outputs, dtype=np.float64), where=(outputs!=0))
    log_2 = np.log2(1-outputs, out=np.zeros_like(outputs, dtype=np.float64), where=(1-outputs!=0))
    return -1.0/outputs.shape[0] * np.sum((log_1 * outputs) + (log_2 * (1 - outputs)), axis=0)


def MI(predictions):
    entropies = entropy(predictions)
    expectations = expectation(predictions)
    return entropies - expectations

def MI_per_class(predictions):
    entropies = entropy_per_class(predictions)
    expectations = expectation_per_class(predictions)
    return entropies - expectations
    

def group_analysis(prediction, uncertainty_map, ground_truth, percentile=90, p=None):
    if p == None:
        p = np.percentile(uncertainty_map.flatten(), percentile)

    mean_predictions = prediction.mean(axis=0)

    certain = (uncertainty_map >= p)
    uncertain = (uncertainty_map < p)

    certain_group_predictions = mean_predictions.copy()
    certain_group_predictions[np.stack([certain] * NEW_CLASSES, axis=0)] = 0
    certain_group_labels = ground_truth.copy()
    certain_group_labels[certain] = 0

    uncertain_group_predictions = mean_predictions.copy()
    uncertain_group_predictions[np.stack([uncertain] * NEW_CLASSES, axis=0)] = 0
    uncertain_group_labels = ground_truth.copy()
    uncertain_group_labels[uncertain] = 0

    # for calculating which classes are mostly filtered out
    # classes_uncertain = []
    # classes_certain = []
    # classes = []
    # argmax_predictions = mean_predictions.argmax(axis=0).flatten()
    # for i in range(NEW_CLASSES):
    #     uncertain_group_classes = uncertain_group_predictions.argmax(axis=0)
    #     certain_group_classes = certain_group_predictions.argmax(axis=0)
    #     group_classes = ground_truth.flatten()
    
    #     if len(argmax_predictions[argmax_predictions == i]) > 0:
    #         classes_uncertain.append(round(len(uncertain_group_classes[uncertain_group_classes == i]) / len(argmax_predictions[argmax_predictions == i]), 5))
    #         classes_certain.append(round(len(certain_group_classes[certain_group_classes == i]) / len(argmax_predictions[argmax_predictions == i]), 5))
    #         classes.append(round(sum(group_classes[group_classes == i]) / len(group_classes), 5))
    #     else:
    #         classes_uncertain.append(0)
    #         classes_certain.append(0)
    #         classes.append(round(sum(group_classes[group_classes == i]) / len(group_classes), 5))

    # print("\t".join([str(x) for x in classes_uncertain]))
    # print("\t".join([str(x) for x in classes_certain]))
    # print("\t".join([str(x) for x in classes]))


    dice_per_class_certain = soft_dice(torch.Tensor(certain_group_predictions.reshape((1, NEW_CLASSES, -1))), torch.Tensor(certain_group_labels.reshape((1, -1))), reduction="dice_per_class", dims=(-1)).detach().cpu().numpy()
    dice_certain = np.nanmean(dice_per_class_certain)

    dice_per_class_uncertain = soft_dice(torch.Tensor(uncertain_group_predictions.reshape((1, NEW_CLASSES, -1))), torch.Tensor(uncertain_group_labels.reshape((1, -1))), reduction="dice_per_class", dims=(-1)).detach().cpu().numpy()
    dice_uncertain = np.nanmean(dice_per_class_uncertain)

    # print("certain / uncertain: ", round(dice_certain, 3), round(dice_uncertain, 3))
    # for i in range(NEW_CLASSES):
    #     print(NEW_CLASS_DICT[i], round(dice_per_class_certain[i], 3), round(dice_per_class_uncertain[i], 3))

    return dice_per_class_certain, dice_per_class_uncertain


def filter_uncertain_images(image_uncertainties, percentile=90, clas=None):
    
    if clas == "lipid":
        p = np.percentile(image_uncertainties[:, 4], percentile)
        uncertain = (image_uncertainties[:, 4] >= p)
    elif clas == "calcium":
        p = np.percentile(image_uncertainties[:, 5], percentile)
        uncertain = (image_uncertainties[:, 5] >= p)
    elif clas == "both":
        p = np.percentile(np.mean(image_uncertainties[:, 4:5], axis=1), percentile)
        uncertain = (np.mean(image_uncertainties[:, 4:5], axis=1) >= p)
    elif clas == "important_structures": # intima, lipid, calcium, media, sidebranch, healed plaque
        important_structures = image_uncertainties[:, 3:6] + image_uncertainties[:, 8:9]
        p = np.percentile(np.mean(important_structures, axis=1), percentile)
        uncertain = (np.mean(important_structures, axis=1) >= p)
    elif clas == "not_per_class":
        p = np.percentile(image_uncertainties, percentile)
        uncertain = (image_uncertainties >= p)
    else: 
        p = np.percentile(image_uncertainties.mean(axis=1), percentile)
        uncertain = (image_uncertainties.mean(axis=1) >= p)

    return uncertain


def filter_uncertain_images_test(image_uncertainties, predictions, percentile=90, clas=None):
    
    # image_uncertainties = (nr_images, nr_classes)
    # argmaxed predictions = (nr_images, height, width)

    #TODO: doesn't work, dice of uncertain images is higher??

    weights = np.zeros(image_uncertainties.shape)
    for c in range(NEW_CLASSES):
        clas = (np.size(predictions[0]) + 1e-6) / (np.sum(np.where(predictions == c, 1, 0), axis=(1,2)) + 1e-6)
        weights[:, c] = clas

    uncertainties_per_class = image_uncertainties * weights
    print(uncertainties_per_class[0])
    print(image_uncertainties[0])

    p = np.percentile(uncertainties_per_class.mean(axis=1), percentile) 
    uncertain = (uncertainties_per_class.mean(axis=1) >= p)

    return uncertain