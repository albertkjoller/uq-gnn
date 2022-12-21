
import torch
from torch.nn import GaussianNLLLoss


class NIGLoss:
    def __init__(self, lambd_) -> None:
        self.lambd_ = lambd_

    def __call__(self, evidential_params_, y, kappa=0):
        """
        Forward pass through the NIGLoss function.

        Parameters
        ----------
        evidential_params_: parameters of the evidential distribution (shape: batch_size x 4)
        y: target attribute of the regression task (shape: batch_size x 1)
        kappa: trade-off parameter between RMSE and NIG loss

        Returns
        -------
        A tuple of the total loss as well as a dictionary of "extra" losses / subparts of the total loss.
        ('Loss', total_loss), {'NLL': nll_loss, 'REG': reg_loss, 'RMSE': rmse_loss}

        """
        # Separate the evidential parameters
        self.gamma, self.nu, self.alpha, self.beta = evidential_params_[:, 0].reshape(-1, 1), \
                                                     evidential_params_[:, 1].reshape(-1, 1), \
                                                     evidential_params_[:, 2].reshape(-1, 1), \
                                                     evidential_params_[:, 3].reshape(-1, 1)

        # Get losses
        nll_loss = self.NIG_NLL(y)
        reg_loss = self.NIG_REGULARIZER(y)
        rmse_loss = torch.sqrt(torch.mean((self.gamma - y) ** 2))

        # Compute total loss
        total_loss = nll_loss + (self.lambd_ * reg_loss)
        total_loss = (1 - kappa) * total_loss.mean() + kappa * rmse_loss
        return ('Loss', total_loss), {'NLL': nll_loss.mean(), 'REG': reg_loss.mean(), 'RMSE': rmse_loss}

    def NIG_NLL(self, y):
        """
        Computes negative log-likelihood of the NIG distribution for a regression target, y. Maximizes model fit.
        Implementation follows Equation 8 in this paper (https://arxiv.org/pdf/1910.02600.pdf)

        """
        omega = 2 * self.beta * (1 + self.nu)
        nll = 0.5 * torch.log(torch.pi / (self.nu)) \
              - self.alpha * torch.log(omega) \
              + (self.alpha + 0.5) * torch.log((y - self.gamma) ** 2 * self.nu + omega) \
              + torch.lgamma(self.alpha) - torch.lgamma(self.alpha + 0.5)
        return nll

    def NIG_REGULARIZER(self, y):
        """
        Computes regularizing loss on the NIG distribution for a regression target, y. Minimizes evidence on errors by
        scaling the error with the total evidence of the infered posterior.
        Implementation follows Equation 9 in this paper (https://arxiv.org/pdf/1910.02600.pdf)

        """
        return abs(y - self.gamma) * (2 * self.nu + self.alpha)


class RMSELoss:
    def __init__(self, ) -> None:
        pass

    def __call__(self, mu, y):
        self.mu = mu.reshape(-1,1)
        self.y = y.reshape(-1,1)

        # Get losses
        rmse_loss = torch.sqrt(torch.mean((self.mu - self.y)**2))
        # Compute total loss
        return ('RMSE', rmse_loss), {}


class GAUSSIANNLLLoss:
    def __init__(self, ) -> None:
        pass

    def __call__(self, theta, y, kappa=0):
        self.mu = theta[:,0].reshape(-1,1)
        self.sigma = theta[:, 1].reshape(-1, 1)
        self.y = y.reshape(-1,1)

        # Compute loss
        loss = GaussianNLLLoss()
        nll_loss = loss(input=self.mu, target=self.y, var=self.sigma)
        rmse_loss = torch.sqrt(torch.mean((self.mu - self.y)**2))

        # Update losses
        #nll_loss = (1 - kappa) * torch.sqrt(nll_loss.mean()) + kappa * rmse_loss
        nll_loss = (1 - kappa) * nll_loss.mean() + kappa * rmse_loss
        return ('GAUSSIANNLL', nll_loss), {'RMSE': rmse_loss}

