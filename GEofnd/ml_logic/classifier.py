#function that splits both categories TRUE or FALSE into 4 Sub categories

def result(prediction, predict_proba):

    if prediction[0] == True:
        if predict_proba[0][1] > 0.70:
            return f"It's true! Probability of {round(predict_proba[0][1]*100,2)}%"
        elif predict_proba[0][1] > 0.5:
            return f'Probably true, probability of {round(predict_proba[0][1]*100,2)}%'
        elif predict_proba[0][1] <= 0.25:
            return f'Seems to be true but lack of info on the subject'
        elif predict_proba[0][1] <= 0.5:
            return f'possibly true dude but probability of {round(predict_proba[0][1]*100,2)}%'

    elif prediction[0] == False:
        if predict_proba[0][0] > 0.70:
            return f"It's a fake news with a probability of {round(predict_proba[0][1]*100,2)}%"
        elif predict_proba[0][0] > 0.5:
            return f'Probably fake with a probability of {round(predict_proba[0][1]*100,2)}%'
        elif predict_proba[0][1] <= 0.25:
            return f'Seems to be fake but lack of info on the subject'
        elif predict_proba[0][1] <= 0.5:
            return f'Possibly fake but probability of {round(predict_proba[0][1]*100,2)}%'
