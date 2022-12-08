#function that splits both categories TRUE or FALSE into 4 Sub categories
from ofnd.ml_logic.params import MODEL_TYPE

def label(prediction_score):
    if MODEL_TYPE == 'ml':
        if prediction_score[0] == 'True':
            if prediction_score[1] > 0.2:
                return 'This article is True'
            else:
                return 'This article is possibly True'
        if prediction_score[0] == 'False':
            if prediction_score[1] < -0.2:
                return 'This article is False'
            else:
                return 'This article is possibly False'

    if MODEL_TYPE == 'tensorflow':
        if prediction_score >= 0.9:
            return 'This article is True'
        if prediction_score >= 0.7 and prediction_score < 0.9:
            return 'This article is possibly True'
        if prediction_score >= 0.5 and prediction_score < 0.7:
            return 'This article is possibly False'
        if prediction_score < 0.5:
            return 'This article is False'

    return 'Something went wrong'
