#function that splits both categories TRUE or FALSE into 4 Sub categories
from ofnd.ml_logic.params import MODEL_TYPE

def label(prediction_score):
    if MODEL_TYPE == 'ml':
        if prediction_score[0] == 'True':
            if prediction_score[1] > 0.2:
                return 'This article is true'
            else:
                return 'This article is probably true'
        if prediction_score[0] == 'False':
            if prediction_score[1] < -0.2:
                return 'This article is False'
            else:
                return 'This article is probably false'

    if MODEL_TYPE == 'tensorflow':
        if prediction_score >= 0.92:
            return 'This article is True'
        if prediction_score >= 0.87 and prediction_score < 0.92:
            return 'This article is probably true'
        if prediction_score >= 0.8 and prediction_score < 0.87:
            return 'This article is probably false'
        if prediction_score < 0.8:
            return 'This article is false'

    return 'Something went wrong'
