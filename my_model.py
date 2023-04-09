import numpy as np
from numpy.random import randn, randint
from utils import load_mnist, to_1_dim, accuracy
class double_Layer_MLP:
    def __init__(self, X, Dhid, Y, X_val, Y_val):
        self.X = X
        self.Y = Y
        self.X_val = X_val
        self.Y_val = Y_val
        self.Din = X.shape[1]
        self.Dout = Y.shape[1]
        self.Dhid = Dhid
        self.W1 = randn(self.Din, self.Dhid)
        self.W2 = randn(self.Dhid, self.Dout)
    
    def act_F(self, M):
        return 1 / (1 + np.exp(-M))
    def train(self, epoch, lr, lam):
        acc_val = []
        loss_train = []
        loss_val = []
        for x in range(epoch):
            for idx in range(self.X.shape[0]):
                idx_val = randint(0, self.Y_val.shape[0])
                Z = self.X_val[idx_val, :] @ self.W1
                H = self.act_F(Z)
                Y_pred = H @ self.W2
                loss_val.append(np.square(Y_pred - self.Y_val[idx_val, :]).mean())

                Z = self.X[idx, :] @ self.W1
                H = self.act_F(Z)
                Y_pred = H @ self.W2
                if not idx % 100:
                    acc_val.append(accuracy(to_1_dim(self.predict(self.X_val)), to_1_dim(self.Y_val)))
                    
                Z = self.X[idx, :] @ self.W1
                H = self.act_F(Z)
                Y_pred = H @ self.W2
                loss_train.append(np.square(Y_pred - self.Y[idx, :]).mean())

                # idx = randint(0, self.X.shape[0])
                dW2 = np.outer(H.T , (Y_pred - self.Y[idx, :])) + lam * self.W2 
                dW1 = np.outer(self.X[idx, :], (((Y_pred - self.Y[idx, :]) @ self.W2.T) * H * (1 - H))) + lam * self.W1
                self.W2 -= lr * dW2
                self.W1 -= lr * dW1
        return acc_val, loss_train, loss_val, self.W1, self.W2


    def predict(self, X):
        Z = X @ self.W1
        H = self.act_F(Z)
        Y_pred = H @ self.W2
        return Y_pred

