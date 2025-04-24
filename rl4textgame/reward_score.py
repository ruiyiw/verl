def reward_func(data_source, solution_str, ground_truth, extra_info):
    if data_source.startswith("textworld"):
        from rl4textgame.tasks.textworld import compute_score
        return compute_score(solution_str, ground_truth, extra_info)
    else:
        raise NotImplementedError