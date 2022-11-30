#function that splits both categories TRUE or FALSE into 4 Sub categories
def labels(score):

    if score > 0.8:
        return "This article is True"
    elif score > 0.5:
        return "It's article is possibly True"
    elif score > 0.2:
        return "This article is False"
    else:
        return "This article is possibly False"
