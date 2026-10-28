import unittest
import numpy as np
import tempfile
from pathlib import Path
import pandas as pd
from types import SimpleNamespace
from pipeline import aggregate, partition, calibrate, prepare

class ProtocolTests(unittest.TestCase):
    def test_csv_protocol_excludes_unknown_and_fits_scaler_on_training_only(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            train = pd.DataFrame({'x': np.arange(90, dtype=float), 'label': np.repeat(['benign','attack','withheld'],30)})
            test = pd.DataFrame({'x': np.arange(1000,1060,dtype=float), 'label': np.repeat(['benign','attack','withheld'],20)})
            train.to_csv(folder/'train.csv',index=False)
            test.to_csv(folder/'test.csv',index=False)
            args = SimpleNamespace(train_csv=str(folder/'train.csv'), test_csv=str(folder/'test.csv'), label='label', unknown='withheld', seed=42)
            x,y,xv,yv,xt,yt,classes,unknown,features,imputer,scaler,source,removed = prepare(args)
            self.assertEqual(classes,['attack','benign'])
            self.assertEqual(len(x)+len(xv),60)
            self.assertEqual(sum(yt==unknown),20)
            self.assertLess(scaler.mean_[0],60)
            self.assertEqual(features,['x'])

    def test_weighted_aggregation(self):
        a = SimpleNamespace(coefs_=[np.array([[1., 3.]])], intercepts_=[np.array([2.])])
        b = SimpleNamespace(coefs_=[np.array([[5., 7.]])], intercepts_=[np.array([6.])])
        result = aggregate([a,b],[1,3])
        np.testing.assert_allclose(result.coefs_[0], [[4.,6.]])
        np.testing.assert_allclose(result.intercepts_[0], [5.])
        np.testing.assert_allclose(a.coefs_[0], [[1.,3.]])

    def test_partition_is_disjoint_complete_reproducible(self):
        y = np.repeat(np.arange(3), 100)
        a, b = partition(y), partition(y)
        np.testing.assert_array_equal(np.sort(np.concatenate(a)), np.arange(300))
        for x,z in zip(a,b): np.testing.assert_array_equal(x,z)

    def test_known_calibration_and_score_direction(self):
        confidence = np.linspace(.5,1,101)
        t = calibrate(confidence,.05)
        self.assertLessEqual(np.mean(confidence<t),.05)
        self.assertTrue(.2<t)

if __name__ == '__main__': unittest.main()
