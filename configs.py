def observation_dim(env: str):
    if 'CartPole' in env:
        return 4
    elif 'FetchReach' in env:
        return 10
    elif 'Fetch' in env:
        return 25
    else:
        raise NotImplementedError


def goal_dim(env: str):
    if 'Fetch' in env:
        return 3
    else:
        raise NotImplementedError


def action_dim(env: str):
    if 'CartPole' in env:
        return 1
    elif 'Fetch' in env:
        return 4
    else:
        raise NotImplementedError


def hidden_size(env: str):
    if 'CartPole' in env:
        return 200
    elif 'Fetch' in env:
        return 600
    else:
        raise NotImplementedError


def num_hidden(env: str):
    if 'CartPole' in env:
        return 1
    elif 'Fetch' in env:
        return 1
    else:
        raise NotImplementedError
