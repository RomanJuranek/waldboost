def takemin(iterable, objective):
    """
    Return an element and its position form an iterable which evaluates
    an objective function to a minimum value among other elements from the
    iterable. This is equivalent to argmin and min functions when the objective
    is simply the value of element. But it can be used in more general settings.

    idx, element = takemin(iterable, objective)

    Example:
    X = np.random.randn(1000)
    idx, value = takemin(X, objective=lambda x: np.abs(x-1))

    This will return index and value of element of X closest to 1.
    """
    take_el = None
    best = None
    best_idx = None
    for idx, el in enumerate(iterable):
        m = objective(el)
        if best is None or m < best:
            best = m
            take_el = el
            best_idx = idx
    return best_idx, take_el


def save_cache(data, filename):
    import pickle
    with open(filename, "wb") as f:
        pickle.dump(data, f)
