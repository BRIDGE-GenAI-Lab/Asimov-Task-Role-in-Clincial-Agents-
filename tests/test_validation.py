from analysis.validation import cohen_kappa, judge_auroc, fleiss_kappa_matrix


def test_cohen_kappa_perfect():
    assert cohen_kappa([1, 0, 1, 1], [1, 0, 1, 1]) == 1.0


def test_judge_auroc_separates():
    truth = [1, 1, 0, 0]
    judge_scores = [0.9, 0.8, 0.2, 0.1]
    assert judge_auroc(truth, judge_scores) == 1.0


def test_fleiss_kappa_runs():
    mat = [[2, 0], [0, 2], [2, 0]]
    k = fleiss_kappa_matrix(mat)
    assert k > 0.8
