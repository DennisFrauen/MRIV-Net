import unittest
from data import sim_gp
import numpy as np

class TestSimGP(unittest.TestCase):
    def test_well_specified(self):
        #np.random.seed(1234567891)

        #Check whether tau is actually the ITE function in noiseless case (no confounding, no error epsilon)
        n = 3000
        data, comp, _ = sim_gp.simulate_data(n=n, p=5, sigma_U=0.2, alpha_U=1, sigma_A=0.2, sigma_Y=0.2, plot=False, scale=False)
        A = data[:, 1]
        Y = data[:, 0]

        Y_cf = comp[6]
        tau = comp[0]
        #Potential outcomes
        Y1 = A*Y + (1-A)*Y_cf
        Y0 = (1-A)*Y + A*Y_cf

        ites = Y1 - Y0
        diff = tau - ites
        self.assertTrue((diff < 0.00001).all())


if __name__ == '__main__':
    unittest.main()