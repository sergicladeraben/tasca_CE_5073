import pickle
from flask import Flask, jsonify, request
from predict_services import logistic_regression_predict, svm_predict, decision_tree_predict, knn_predict

app = Flask('model-predict')

class_mapping = {
    0: 'Iris Setosa',
    1: 'Iris Versicolour',
    2: 'Iris Virginica',
}

@app.route('/regressionpredict', methods=['POST'])
def regressionPredict():
    
    with open('models/logistic_regression_model.pkl', 'rb') as f:
        scaler, model = pickle.load(f)
    
    customer = request.get_json()
    class_pred = logistic_regression_predict(customer, scaler, model)

    result = {
        'Flor': class_mapping[int(class_pred)],
    }

    return jsonify(result)

@app.route('/svmpredict', methods=['POST'])
def svmPredict():
    with open('models/svm_model.pkl', 'rb') as f:
        scaler, model = pickle.load(f)

    customer = request.get_json()
    class_pred = svm_predict(customer, scaler, model)

    result = {
        'Flor': class_mapping[int(class_pred)],
    }

    return jsonify(result)

@app.route('/decisiontreepredict', methods=['POST'])
def decisionTreePredict():
    with open('models/decision_tree_model.pkl', 'rb') as f:
        scaler, model = pickle.load(f)

    customer = request.get_json()
    class_pred = decision_tree_predict(customer, scaler, model)

    result = {
        'Flor': class_mapping[int(class_pred)],
    }

    return jsonify(result)


@app.route('/knnpredict', methods=['POST'])
def knnPredict():
    with open('models/knn_model.pkl', 'rb') as f:
        scaler, model = pickle.load(f)

    customer = request.get_json()
    class_pred = knn_predict(customer, scaler, model)

    result = {
        'Flor': class_mapping[int(class_pred)],
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=8000)  