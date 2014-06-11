import preprocess
import numpy
from numpy import linalg

class TestPreprocess():
    def setUp(self):
        pass

    def testSigmoid(self):
        self.assertAlmostEqual(0.5, preprocess.sigmoid(0))
        self.assertAlmostEqual(0.7310585786300049, preprocess.sigmoid(1))
        self.assertAlmostEqual(0.2689414213699951, preprocess.sigmoid(-1))
        
        t = numpy.array([-2, -1, 0, 1, 2])
        expected = numpy.array([0.11920292202211755, 0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823])
        actual = preprocess.sigmoid(t)
        self.assertAlmostEqual(0.0, linalg.norm(expected - actual))
    
    def testNormalize(self):
        def test(x):
            mu, sigma = preprocess.muSigma(x)

            self.assertAlmostEqual(1.23902738264240, x[1][2])
            
            self.assertEqual(5, len(mu))
            self.assertEqual(5, len(sigma))
            
            self.assertAlmostEqual(2.87969736221038, mu[0])
            self.assertAlmostEqual(2.04868506865762, sigma[0])
            self.assertAlmostEqual(-0.99025024303433, (x[0][0] - mu[0]) / sigma[0])
            
            self.assertAlmostEqual(1.97861578296198, mu[2])
            self.assertAlmostEqual(2.33076030134340, sigma[2])
            self.assertAlmostEqual(-0.31731637092553, (x[1][2] - mu[2]) / sigma[2])
            
            y = preprocess.normalize(x, mu, sigma)
            
            m, n = y.shape
            self.assertEqual(4, m)
            self.assertEqual(5, n)
            
            self.assertAlmostEqual(-0.99025024303433, y[0][0])
            self.assertAlmostEqual(-0.31731637092553, y[1][2])
            
            u = preprocess.sigmoid(y)
            self.assertAlmostEqual(0.27086265279957, u[0][0])
            self.assertAlmostEqual(0.42132990768430, u[1][2])
                    
        x = [[0.85098647507137, 1.52729384756342, 0.25020360869713, 8.28459941133853, 3.51078305926006],
             [2.04372712000375, 6.94129011291125, 1.23902738264240, 2.19309403749903, 9.52302982804621],
             [2.32826261875266, 3.94437614959272, 5.96487525728262, 3.75875046247617, 8.92013842421705],
             [6.29581323501374, 5.99925219683617, 0.46035688322576, 7.78675934506873, 8.84119588325795]]
        xp = [numpy.array(row) for row in x]
        xpp = numpy.array(x)
        test(x)
        test(xp)
        test(xpp)
        
        
