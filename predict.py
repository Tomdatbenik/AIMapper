import numpy as np
from bag_and_bow import bag

def get_result(result, labels):
    ERROR_THRESHOLD = 0.1

    results = [[i, r] for i, r in enumerate(result) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        label = labels[r[0]]
        return_list.append(
            {"intent": label, "probability": str(r[1])})

    return return_list

def predict(text:str, prediction_model, labels):
    prediction = prediction_model.predict(np.array([bag(text)]))[0]
    
    readable_prediction = get_result(prediction, labels)

    for result in readable_prediction:
        if result['intent'] in labels:
            return {"prediction": readable_prediction}

    return {"prediction": readable_prediction}