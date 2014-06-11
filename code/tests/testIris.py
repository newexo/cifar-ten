import dataset

class TestIris():

    def setUp(self):
        pass

    def testIris(self):
        data = dataset.Iris()
        
        # there should be the same number of labels as datapoints in each subset
        self.assertEqual(len(data.train[0]), len(data.train[1]))
        self.assertEqual(len(data.test[0]), len(data.test[1]))
        self.assertEqual(len(data.valid[0]), len(data.valid[1]))
        
        # there should be 150 datapoints in total
        self.assertEqual(150, len(data.train[0]) + len(data.test[0]) + len(data.valid[0]))
        
