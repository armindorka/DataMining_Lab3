import numpy as np


def z_normalize(ts):
    mean = np.mean(ts)
    std = np.std(ts)
    return (ts - mean) / std if std > 0 else np.zeros_like(ts)


def compute_distance_profile(ts, query):
    m = len(query)
    dp_len = len(ts) - m + 1
    dp = np.empty(dp_len)
    query = z_normalize(query)

    for i in range(dp_len):
        dp[i] = np.linalg.norm(z_normalize(ts[i : i + m]) - query)

    return dp
 
# m: length of querry
def apply_exclusion(dp, index, m, exclude):
    # TODO: apply the exclusion zone around the indices in dp
    size_of_exclusion = int(m * exclude // 2)
    for i in range(index - size_of_exclusion, index + size_of_exclusion + 1):
    # We have to set the same ones to infinitiy, since we are looking for the minimum distance
        if i >= 0 and i < len(dp):
            dp[i] = np.inf


def matrix_profile(ts, m, exclude=0.0):

    n = len(ts)
    mp = np.full(n - m + 1, np.inf) # matrix profile
    mp_idx = np.full(n - m + 1, -1) # matrix profile index

    for i in range(n - m + 1):
        query = ts[i : i + m]
        dp = compute_distance_profile(ts, query)
        apply_exclusion(dp, i, m, exclude)

        min_idx = np.argmin(dp)
        mp_idx[i] = min_idx
        mp[i] = dp[min_idx]

    return mp, mp_idx


def find_discords(ts, window, k):
    """Detects the top-k discords in a time series."""
    mp, mp_idx = matrix_profile(ts, window)
    discords = []

    for _ in range(k):
        # TODO 1) Find the index of the highest Matrix Profile value
        max_idx = np.argmax(mp)
        discords.append(max_idx)
        # TODO 2) Apply exclusion zone to prevent redundant discords
        # Remember that you can exclude points with -np.inf, since
        # we are searching for the largest values.
        
        for i in range(max_idx, max_idx + window + 1):
        # We have to set the same ones to infinitiy, since we are looking for the minimum distance
            if i >= 0 and i < len(ts):
                mp[i] = -np.inf

    return discords
