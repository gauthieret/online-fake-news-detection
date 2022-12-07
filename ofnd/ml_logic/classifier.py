#function that splits both categories TRUE or FALSE into 4 Sub categories
def label(prediction_score):
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
