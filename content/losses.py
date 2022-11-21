
import torch

class NIGLoss:
    def __init__(self, lambd_) -> None:
        self.lambd_ = lambd_

    def __call__(self, evidential_params_, y):
        self.gamma, self.v, self.alpha, self.beta = evidential_params_[:, 0].reshape(-1, 1), \
                                                    evidential_params_[:, 1].reshape(-1, 1), \
                                                    evidential_params_[:, 2].reshape(-1, 1), \
                                                    evidential_params_[:, 3].reshape(-1, 1)

        # Get losses
        nll_loss = self.NIG_NLL(y)
        reg_loss = self.NIG_REGULARIZER(y)

        # Compute total loss
        total_loss = nll_loss + (self.lambd_ * reg_loss)
        return ('Loss', total_loss.mean()), (('NLL', nll_loss.mean()), ('REG', reg_loss.mean()))

    def NIG_NLL(self, y, eps=1e-7):
        """
        Computes negative log-likelihood of the NIG distribution for a regression target, y. Maximizes model fit.
        Implementation follows Equation 8 in this paper (https://arxiv.org/pdf/1910.02600.pdf)
        """

        omega = 2 * (self.beta + eps) * (1+self.v)
        nll = 0.5 * torch.log(torch.pi / (self.v + eps)) \
              - self.alpha * torch.log(omega) \
              + (self.alpha + 0.5) * torch.log((y-self.gamma)**2*(self.v + eps)+omega) \
              + torch.lgamma(self.alpha) - torch.lgamma(self.alpha + 0.5)

        return nll

    def NIG_REGULARIZER(self, y, eps=1e-7):
        """
        Computes regularizing loss on the NIG distribution for a regression target, y. Minimizes evidence on errors by
        scaling the error with the total evidence of the infered posterior.
        Implementation follows Equation 9 in this paper (https://arxiv.org/pdf/1910.02600.pdf)
        """
        reg_loss = abs(y - self.gamma) * (2 * (self.v + eps) + self.alpha)
        return reg_loss


class MSELoss:

    def __init__(self, ) -> None:
        pass

    def __call__(self, evidential_params_, y):
        self.gamma, self.v, self.alpha, self.beta = evidential_params_[:, 0].reshape(-1, 1), \
                                                    evidential_params_[:, 1].reshape(-1, 1), \
                                                    evidential_params_[:, 2].reshape(-1, 1), \
                                                    evidential_params_[:, 3].reshape(-1, 1)
        # Get losses
        mse_loss = (self.gamma - y)**2
        # Compute total loss
        return ('MSE', mse_loss.mean()), None

