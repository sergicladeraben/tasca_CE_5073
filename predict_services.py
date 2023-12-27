def logistic_regression_predict(customer, scaler, model):
    petal_length = float(customer['petal_length'])
    petal_width = float(customer['petal_width'])
    
    x = scaler.transform([[petal_length, petal_width]])
    class_pred = model.predict(x)[0]
    return class_pred

def svm_predict(customer, scaler, model):
    petal_length = float(customer['petal_length'])
    petal_width = float(customer['petal_width'])
    
    x = scaler.transform([[petal_length, petal_width]])
    class_pred = model.predict(x)[0]
    return class_pred

def decision_tree_predict(customer, scaler, model):
    petal_length = float(customer['petal_length'])
    petal_width = float(customer['petal_width'])
    
    x = scaler.transform([[petal_length, petal_width]])
    class_pred = model.predict(x)[0]
    return class_pred

def knn_predict(customer, scaler, model):
    petal_length = float(customer['petal_length'])
    petal_width = float(customer['petal_width'])
    
    x = scaler.transform([[petal_length, petal_width]])
    class_pred = model.predict(x)[0]
    return class_pred