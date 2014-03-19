import test_common
from spartan.examples.sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits

N_TREES = 50

class TestRandomForest(test_common.ClusterTest):
  def test_forest(self):
    ds = load_digits()
    X = ds.data
    y = ds.target
    rf = RandomForestClassifier(n_estimators = N_TREES)
    rf.fit(X, y)
    #make sure it memorize data
    assert rf.score(X, y) >= 0.95
