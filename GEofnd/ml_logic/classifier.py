#function that splits both categories TRUE or FALSE into 4 Sub categories

def result(prediction, predict_proba):

    if prediction[0] == True:
        if predict_proba[0][1] > 0.70:
            return "It's true!"
        elif predict_proba[0][1] > 0.5:
            return 'Probably true'
        elif predict_proba[0][1] <= 0.5:
            return 'possibly true dude'

    elif prediction[0] == False:
        if predict_proba[0][0] > 0.70:
            return "It's a fake news"
        elif predict_proba[0][0] > 0.5:
            return 'Probably fake'
        elif predict_proba[0][1] <= 0.5:
            return 'Possibly fake'
