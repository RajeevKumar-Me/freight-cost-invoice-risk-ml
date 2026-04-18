from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error , mean_absolute_error , r2_score

def Train_linear_regression(X_train , Y_train):
    model = LinearRegression()
    model.fit(X_train , Y_train)
    return model

def Train_Decison_tree(X_train , Y_train , max_depth = 4):
    model = DecisionTreeRegressor(max_depth = max_depth , random_state = 42)
    model.fit (X_train, Y_train)
    return model

def Train_random_forest (X_train , Y_train, max_depth =4):
    model = RandomForestRegressor(max_depth=4 , random_state = 42)
    model.fit( X_train , Y_train)
    return model
 
def evaluate_model (model , X_test , Y_test , model_name: str):
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(Y_test , preds)
    mae = mean_absolute_error(Y_test , preds)
    r2 = r2_score(Y_test , preds) * 100

    print(f"\n{model_name} Performance:")
    print (f"MAE : {mae :.2f}")
    print(f"RMSE : {rmse :.2f}")
    print(f"R2 : {r2 :.2f}%")

    return {
      "model_name": model_name,
      "mae": mae,
      "rmse": rmse,
      "r2": r2
    }
    