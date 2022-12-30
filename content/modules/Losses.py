
import torch
from torch.nn import GaussianNLLLoss


class NIGLoss:
    def __init__(self, lambd_) -> None:
        self.lambd_ = lambd_
        self.scalar = None

    def __call__(self, evidential_params_, y, kappa=0, training=True):
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
        y = y.reshape(-1, 1)

        # if training on scaled data
        if self.scalar is not None:
            # if in eval, then models outputs descaled data
            if training: # then scale target variable

                # de-scale y for NLL loss
                nll_loss = self.NIG_NLL(torch.from_numpy(self.scalar.transform(y)))
                reg_loss = self.NIG_REGULARIZER(torch.from_numpy(self.scalar.transform(y)))
                # de-scale gamma for RMSE
                rmse_loss = torch.sqrt(torch.mean((torch.from_numpy(self.scalar.inverse_transform(self.gamma.detach())) - y) ** 2))

                # Compute total loss
                total_loss = nll_loss + (self.lambd_ * reg_loss)
                total_loss = (1 - kappa) * total_loss.mean() + kappa * rmse_loss

            else: # not training and model output is descaled
                # rmse is as normal (model output has been converted)
                rmse_loss = torch.sqrt(torch.mean((self.gamma - y) ** 2))
                # NLL has to be scaled on both terms
                y = torch.from_numpy(self.scalar.transform(y))
                nll_loss = self.NIG_NLL(y)
                reg_loss = self.NIG_REGULARIZER(y)

        else:
            # Compute loss like normal
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
        # loss function
        self.loss_func = GaussianNLLLoss(reduction='mean')
        self.scalar = None

    def __call__(self, theta, y, kappa=0, training=True):
        mu = theta[:,0].reshape(-1,1)
        var = theta[:, 1].reshape(-1, 1)
        y = y.reshape(-1, 1)

        # if training on scaled data
        if self.scalar is not None:
            # if in eval, then models outputs descaled data
            if training: # then scale target variable
                # de-scale target for NLL loss
                nll_loss = self.loss_func(target=torch.from_numpy(self.scalar.transform(y)), input=mu, var=var)
                # de-scale prediction for RMSE
                rmse_loss = torch.sqrt(torch.mean((torch.from_numpy(self.scalar.inverse_transform(mu.detach())) - y) ** 2))

            else: # not training and model output is descaled
                # rmse is as normal (model output has been converted)
                rmse_loss = torch.sqrt(torch.mean((mu - y) ** 2))
                # NLL has to be scaled on both terms
                y = torch.from_numpy(self.scalar.transform(y))
                # scaling for NLL loss
                mu = torch.from_numpy(self.scalar.transform(mu))
                var = var/self.scalar.var_
                nll_loss = self.loss_func(target=y, input=mu, var=var)

        else:
            # Compute loss like normal
            nll_loss = self.loss_func(target=y, input=mu, var=var)
            rmse_loss = torch.sqrt(torch.mean((mu - y)**2))


        # kappa automatically decays if defined
        #   - loss is Gaussian NLL if kappa is not defined
        loss = (1 - kappa) * nll_loss + kappa * rmse_loss

        return ('Loss', loss), {'GAUSSIANNLL': nll_loss, 'RMSE': rmse_loss}



