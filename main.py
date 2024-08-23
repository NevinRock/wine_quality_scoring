from ucimlrepo import fetch_ucirepo
import datetime
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, f1_score
import numpy as np


def predict_scoreing(y_true: pd.DataFrame, y_pred: pd.DataFrame):
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)

    # Calculate Range-MAE (Mean Absolute Error within the range [0, 30])
    mask = (y_true <= 30)  # Create a mask for true values within the range [0, 30]
    range_mae = mean_absolute_error(y_true[mask], y_pred[mask])  # Calculate MAE for predictions within the range

    # Calculate F1-score (binary classification within the range [0, 30])
    y_true_binary = (y_true <= 30).astype(int)  # Convert true values to binary (1 if <=30, else 0)
    y_pred_binary = (y_pred <= 30).astype(int)  # Convert predicted values to binary (1 if <=30, else 0)
    f1 = f1_score(y_true_binary, y_pred_binary)  # Calculate F1-score based on binary classification

    # Calculate the final score using the given formula
    score = 0.5 * (1 - mae / 100) + 0.5 * f1 * (1 - range_mae / 100)  # Combine MAE, Range-MAE, and F1-score

    # Print the results
    print(f"MAE: {mae:.2f}, Range-MAE: {range_mae:.2f}, F1: {f1:.2f}, Final Score: {score:.2f}")

# Define a callback function to print the results of the validation set
def print_validation_result(env: pd.Series):
    result = env.evaluation_result_list[-1]
    print(f"[{env.iteration}] {result[1]}'s {result[0]}: {result[2]}")


if __name__ == '__main__':
    wine_quality = fetch_ucirepo(id=186)

    # data (as pandas dataframes)
    X = wine_quality.data.features
    y = wine_quality.data.targets

    # concate ft and targ
    df = pd.concat([X, y], axis=1)

    df_output = df

    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:, :-1],
        df.iloc[:, -1],
        test_size=0.2,
        random_state=42
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)



    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "root_mean_squared_error",
        "max_depth": 7,
        "learning_rate": 0.02,
        "verbose": 0,
    }

    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=30000,
        valid_sets=[test_data],
        callbacks=[print_validation_result],
    )

    # prediction = gbm.predict(df.iloc[:, :-1]).astype(int)

    prediction = np.round(gbm.predict(df.iloc[:, :-1]))

    df_output["quality_prediction"] = prediction
    df_output.to_csv(("output/output_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"), header=None, index=False)

    predict_scoreing(df_output["quality"], df_output["quality_prediction"])
