import numpy as np

class LinearRegressionCustom:

    def __init__(self, lr=0.01, rp=0, epochs=1000, patience=10, verbose=True):
        self.lr = lr
        self.rp = rp
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose

    def fit(self, X, y, X_val, y_val):
        m, n = X.shape
        self.w = np.zeros(n)

        best_loss = float('inf')
        wait = 0

        for epoch in range(1, self.epochs + 1):
        
            ## Try to improve result by reducing numercial round off error 
            # # Gradient Descent
            # self.w -= self.lr * ((1 / m) * (X.T @ ((X @ self.w) - y.values)))

            # # Losses
            # train_loss = (((X @ self.w) - y.values) ** 2).mean()
            # val_loss = ((X_val @ self.w - y_val.values) ** 2).mean()

            # Predicting
            preds = X @ self.w
            error = preds - y.values

            # Gradient
            grad = (1 / m) * (X.T @ error)

            # Gradient Descent 
            self.w -= (self.lr * (grad+ (self.rp*self.w))) ## Along with regularization(with 0<rp<=0.00001, it have same result as when rp = 0) 

            # Losses
            train_loss = (error ** 2).mean()
            val_loss = ((X_val @ self.w - y_val.values) ** 2).mean()

            # Print progress
            if self.verbose:
                print(
                    f"Epoch {epoch:4d} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f}"
                )

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                best_w = self.w.copy()
                wait = 0
            else:
                wait += 1

            if wait >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        self.w = best_w

    def predict(self, X):
        return X @ self.w
