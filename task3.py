import numpy as np


def create_lagged_features(x, exogenous, lag=3):
    # x is a (n_timestep, ) array
    # exogenous is an (n_timestep, n_variables) array
    X, y = [], []
    for i in range(len(x) - lag):
        # TODO: ensure that the data contains also all exogenous variables.
        # 'concatenate' the lagged feature and the exogenous features.
        y.append(x[i + lag])

        safe_lag = x[i:i + lag]
        if i < len(exogenous):
            X.append(np.concatenate([safe_lag, exogenous[i]]))
        else : # We take the value before
            X.append(np.concatenate([safe_lag, exogenous[len(exogenous) - 1]]))

    return np.asarray(X), np.asarray(y)


def iterative_forecast(reg, last_known, last_known_exogenous, steps):
    # last_known_exogenous is the last known exogenous variable of the training data
    # last_known is the last_known lagged training sample
    forecast = []
    window = list(last_known)
    for _ in range(steps):
        # TODO: ensure the predict function also gets the exogenous variables
        # Remember that they should occupy the same position in the feature
        # vector
        # Concat the last known with the last exogenous
        concat_known = np.concatenate([window, last_known_exogenous.flatten()])

        #Now, predict:
        predict_from_known = reg.predict(concat_known.reshape(1, -1))

        # Adding to the forecast
        forecast.append(predict_from_known[0])

        # Update rolling window
        window.pop(0)  # Remove oldest value
        window.append(predict_from_known[0])  # Add new predict

    return np.array(forecast)
