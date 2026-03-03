"""
Basic tests for OFMTL-AL implementation
"""

import unittest
import numpy as np
import sys
sys.path.append('..')
from code.ofmtl_al import OFMTLAL, AdaptiveLossFunction
from code.adaptive_loss import adaptive_weights_numpy, adaptive_loss

class TestAdaptiveLoss(unittest.TestCase):
    """Test adaptive loss functions"""
    
    def test_adaptive_weights_bounds(self):
        """Test that adaptive weights are bounded"""
        test_values = np.array([0.1, 0.5, 1.0, 5.0, 10.0])
        tau = 1.0
        
        for val in test_values:
            f2 = np.array([val])
            d = adaptive_weights_numpy(f2, tau)
            
            # Check bounds from Lemma 1
            lower_bound = (1 + tau) / (2 * (val + tau))
            upper_bound = (1 + tau) / (2 * tau)
            
            self.assertGreaterEqual(d, lower_bound)
            self.assertLessEqual(d, upper_bound)
    
    def test_adaptive_loss_properties(self):
        """Test properties of adaptive loss"""
        test_values = [0.1, 1.0, 10.0]
        tau = 1.0
        
        for x in test_values:
            loss = adaptive_loss(np.array([x]), tau)
            self.assertGreaterEqual(loss, 0)  # Non-negative
            self.assertIsInstance(loss, float)

class TestOFMTLAL(unittest.TestCase):
    """Test OFMTL-AL algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 20)
        self.y = np.random.randint(0, 2, 100)
        self.X_test = np.random.randn(50, 20)
        self.y_test = np.random.randint(0, 2, 50)
        self.model = OFMTLAL(lambda_reg=0.1, tau=1.0)
    
    def test_initialization(self):
        """Test initialization"""
        self.assertEqual(self.model.lambda_reg, 0.1)
        self.assertEqual(self.model.tau, 1.0)
        self.assertEqual(len(self.model.devices), 0)
    
    def test_add_device(self):
        """Test adding a device"""
        error, time_taken = self.model.add_device(
            self.X[:50], self.y[:50], 
            self.X_test, self.y_test
        )
        
        self.assertIsInstance(error, float)
        self.assertIsInstance(time_taken, float)
        self.assertGreaterEqual(error, 0)
        self.assertLessEqual(error, 1)
        self.assertEqual(len(self.model.devices), 1)

if __name__ == '__main__':
    unittest.main()
