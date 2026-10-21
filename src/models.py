"""The five mandatory learners. Always returns FRESH instances (no state shared between runs)."""
from river import ensemble, forest, linear_model, naive_bayes, optim, tree


def make_models(cfg: dict) -> dict:
    m = cfg["models"]
    seed = cfg["protocol"]["seed"]
    h = m["hoeffding"]
    n = m["ensemble_size"]

    def hoeffding():
        return tree.HoeffdingTreeClassifier(
            grace_period=h["grace_period"], delta=h["delta"], tau=h["tau"], leaf_prediction="mc"
        )

    return {
        "Gaussian Naive Bayes": naive_bayes.GaussianNB(),
        "Online Logistic Regression": linear_model.LogisticRegression(optimizer=optim.SGD(m["logistic_lr"])),
        "Hoeffding Tree": hoeffding(),
        "Online Bagging": ensemble.BaggingClassifier(model=hoeffding(), n_models=n, seed=seed),
        "Adaptive Random Forest": forest.ARFClassifier(
            n_models=n, max_features="sqrt", lambda_value=m["arf_lambda"], leaf_prediction="mc",
            grace_period=h["grace_period"], delta=h["delta"], tau=h["tau"], seed=seed,
        ),
    }