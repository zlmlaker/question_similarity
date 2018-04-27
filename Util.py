from keras import backend as K
import math
import numpy as np
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

def precision(y_true,  y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def normalize(vec):
    sum =  np.linalg.norm(vec)
    return vec / sum

def cal_sim(v1, v2):
    sum = np.vdot(v1,v2)
    cos = sum / ( np.linalg.norm(v1) * np.linalg.norm(v2))
    sim = 0.5 + cos * 0.5
    return sim

def precision_np(y_true, y_pred):
    true_positives = np.sum(y_true * y_pred)
    predicted_positives = np.sum(y_pred)
    precision = true_positives / predicted_positives
    return precision

def recall_np(y_true, y_pred):
    true_positives =  np.sum(y_true * y_pred)
    possible_positives = np.sum(y_true)
    recall = true_positives / possible_positives
    return recall

def f1_np(precision, recall):
    return 2 * ((precision * recall) / (precision + recall))
