class ScoreCallback:
    def __init__(self, estimator, X, y, score_function):
        self.estimator = estimator
        self.X = X
        self.y = y
        self.score_function = score_function
        self.n_iter_ = []
        self.scores_ = []

    def __call__(self, n_iter):
        preds = self.estimator.predict(self.X)
        scores = {}
        study_scores = {}
        for study in self.y:
            scores[study] = self.score_function(preds[study]['contrast'],
                                                self.y[study]['contrast'])
        self.n_iter_.append(n_iter)
        self.scores_.append(scores)
        scores_str = ' '.join('%s: %.3f' % (study, score)
                              for study, score in scores.items())
        scores_str = 'Score: ' + scores_str
        print(scores_str)

        scores_str = ' '.join('%s: %.3f' % (study, score)
                              for study, score in study_scores.items())
        scores_str = 'Study score: ' + scores_str
        print(scores_str)


class MultiCallback:
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def __call__(self, *args, **kwargs):
        for name, callback in self.callbacks.items():
            print('[callback %s]' % name)
            callback(*args, **kwargs)
