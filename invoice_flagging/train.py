from data_preprocessing import load_invoice_data , split_data , scale_features , apply_labels
from modeling_evaluation import train_random_forest , evaluate_classifier
import joblib

features =[
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

target = "flag_invoice"

def main():
    df = load_invoice_data()
    df = apply_labels(df)


    X_train , X_test , Y_train , Y_test = split_data(df, features , target)
    X_train_scaled, X_test_scaled = scale_features(X_train , X_test , 'models/scaler.pkl')

    grid_search = train_random_forest(X_train_scaled , Y_train)
    evaluate_classifier(grid_search.best_estimator_ , X_test_scaled , Y_test, "Random Forest Classifier")

    joblib.dump(grid_search.best_estimator_, 'models/predict_flag_invoice.pkl')

if __name__ == "__main__":
    main() 
