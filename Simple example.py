"""
Simple example demonstrating OFMTL-AL usage
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sys
sys.path.append('..')
from code.ofmtl_al import OFMTLAL

def generate_synthetic_data(n_samples=1000, n_features=20, n_classes=2):
    """Generate synthetic dataset"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_state=42
    )
    return X, y

def main():
    print("=" * 50)
    print("OFMTL-AL Simple Example")
    print("=" * 50)
    
    # Generate synthetic data
    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Initialize OFMTL-AL
    model = OFMTLAL(lambda_reg=0.1, tau=1.0)
    
    # Add first device
    print("\nAdding Device 1...")
    error1, time1 = model.add_device(X_train[:300], y_train[:300], X_test, y_test)
    print(f"  Error: {error1:.4f}, Time: {time1:.2f}s")
    
    # Add second device
    print("\nAdding Device 2...")
    error2, time2 = model.add_device(X_train[300:600], y_train[300:600], X_test, y_test)
    print(f"  Error: {error2:.4f}, Time: {time2:.2f}s")
    
    # Add third device
    print("\nAdding Device 3...")
    error3, time3 = model.add_device(X_train[600:900], y_train[600:900], X_test, y_test)
    print(f"  Error: {error3:.4f}, Time: {time3:.2f}s")
    
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"Average error: {(error1 + error2 + error3)/3:.4f}")
    print(f"Average time: {(time1 + time2 + time3)/3:.2f}s")
    print("=" * 50)

if __name__ == "__main__":
    main()
