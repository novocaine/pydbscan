"""
Very basic integration test suite for dbscan, just verifies basic 
clustering of some very simple two-dimensional data.

This could be improved greatly.
"""

from __future__ import print_function
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from nose.tools import assert_equal
import numpy as np
import dbscan
import abc
import subprocess
import tempfile
import os.path


class DbScanBase(object):
    """ 
    Abstract base; concrete classes override how the dbscan onbject is
    created so we can run the same tests on the CLI and python bindings.
    """

    __metaclass__ = abc.ABCMeta

    EXPECTED_NUM_CLUSTERS = 3
    # should be between 0 and 1, values near 1 seem to work well on this data;
    COSINE_EPS = 0.99
    # can be anything
    EUCLIDEAN_EPS = 0.3
    MIN_PTS = 10

    @abc.abstractmethod
    def _create_dbscan(self, sample_data, *args):
        pass

    def setup(self):
        self.sample_data_double = self._generate_sample_data()

    @property
    def sample_data_single(self):
        return self.sample_data_double.astype(np.float32)

    def _generate_sample_data(self):
        """
        generate 3 separate blobs which we expect to identify as separate
        clusters; this is taken from an sklearn example.

        TODO - to reduce dev dependencies, distribute this as static data
        rather than having to use sklearn to gen the data
        """
        centers = [[1, 1], [-1, -1], [1, -1]]

        # note that specifying the same random_state every time means we get the
        # same data every time
        X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                                    random_state=0)
        return StandardScaler().fit_transform(X)

    def _num_clusters(self, labels):
        # -1 means 'not in any cluster', any other integer is a cluster id
        return len(set(labels)) - (1 if -1 in labels else 0)

    def test_nonsparse_single(self):
        """ 
        Non-sparse array with single precision floats, Euclidean distance
        """

        labels = self._create_dbscan(self.sample_data_single, "nonsparse",
            "euclidean").run(self.EUCLIDEAN_EPS, self.MIN_PTS)
        assert_equal(self._num_clusters(labels), self.EXPECTED_NUM_CLUSTERS)

    def test_sparse_single_euclidean(self):
        """
        Sparse array with single precision floats, Euclidean distance metric
        """
        labels = self._create_dbscan(self.sample_data_single, 
                "sparse", "euclidean").run(self.EUCLIDEAN_EPS, self.MIN_PTS)
        assert_equal(self._num_clusters(labels), self.EXPECTED_NUM_CLUSTERS)

    def test_sparse_single_cosine(self):
        """
        Sparse array with single precision floats, cosine similarity metric
        """
        labels = self._create_dbscan(self.sample_data_single, 
                "sparse", "cosine").run(self.COSINE_EPS, self.MIN_PTS)
        assert_equal(self._num_clusters(labels), self.EXPECTED_NUM_CLUSTERS)

    def test_nonsparse_double(self):    
        """ 
        Non-sparse array with double precision floats, Euclidean distance
        """
        labels = self._create_dbscan(self.sample_data_double, "nonsparse",
            "euclidean").run(self.EUCLIDEAN_EPS, self.MIN_PTS)
        assert_equal(self._num_clusters(labels), self.EXPECTED_NUM_CLUSTERS)

    def test_sparse_double_euclidean(self):
        """
        Sparse array with double precision floats, Euclidean distance metric
        """
        labels = self._create_dbscan(self.sample_data_double, 
                "sparse", "euclidean").run(self.EUCLIDEAN_EPS, self.MIN_PTS)
        assert_equal(self._num_clusters(labels), self.EXPECTED_NUM_CLUSTERS)

    def test_sparse_double_cosine(self):
        """
        Sparse array with double precision floats, cosine similarity metric
        """
        labels = self._create_dbscan(self.sample_data_double, 
                "sparse", "cosine").run(self.COSINE_EPS, self.MIN_PTS)
        assert_equal(self._num_clusters(labels), self.EXPECTED_NUM_CLUSTERS)


class TestPyDbScan(DbScanBase):
    """ 
    Integration tests for the python bindings
    """
    def _create_dbscan(self, sample_data, *args):
        return dbscan.dbscan(sample_data, *args)


class TestCLIDbScan(DbScanBase):
    """ 
    Integration tests for the command line interface, 'dbscan' needs to
    be visible in the PATH otherwise the tests will be skipped
    """

    class CLIDbScan(object):
        def __init__(self, data, *args):
            inp = tempfile.NamedTemporaryFile(delete=False)
            for row in data:
                print(",".join(str(s) for s in row), file=inp)
            inp.close()
            self.inp_name = inp.name
            self.args = list(args)
            if data.dtype == np.float32:
                self.precision = "single"
            else:
                self.precision = "double"

        def run(self, eps, min_pts):
            args = [self.dbscan_path(), str(eps), 
                str(min_pts)] + self.args + [self.precision, self.inp_name]
            output = subprocess.check_output(args)
            print(args)
            result = []
            for line in output.split("\n"):
                if line:
                    result.append(int(line))
            return result

        def dbscan_path(self):
            return os.path.join(os.path.dirname(__file__), "..", "dbscan")
                        
    def _create_dbscan(self, data, *args):
        return self.CLIDbScan(data, *args)
