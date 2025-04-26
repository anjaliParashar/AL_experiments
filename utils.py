import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import IdentityMCObjective
from botorch.utils.sampling import sample_hypersphere
from botorch.utils.transforms import t_batch_mode_transform
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from botorch.models import SingleTaskVariationalGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import RBFKernel, ScaleKernel
from torch.quasirandom import SobolEngine
from botorch.models.transforms.outcome import Standardize
from botorch.models import ModelListGP, SingleTaskGP
from gpytorch.constraints import Interval
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
import os
from scipy.spatial.distance import cdist
from torch.optim import Adam

def smooth_mask(x, a, eps=2e-3):
    """Returns 0ish for x < a and 1ish for x > a"""
    return torch.nn.Sigmoid()((x - a) / eps)


def smooth_box_mask(x, a, b, eps=2e-3):
    """Returns 1ish for a < x < b and 0ish otherwise"""
    return smooth_mask(x, a, eps) - smooth_mask(x, b, eps)


class ExpectedCoverageImprovement(MCAcquisitionFunction):
    def __init__(
        self,
        model,
        constraints,
        punchout_radius,
        bounds,
        train_inputs,
        num_samples=128,
        lambda_=0.0,
        **kwargs,
    ):
        """Expected Coverage Improvement (q=1 required, analytic)

        Right now, we assume that all the models in the ModelListGP have
        the same training inputs.

        Args:
            model: A ModelListGP object containing models matching the corresponding constraints.
                All models are assumed to have the same training data.
            constraints: List containing 2-tuples with (direction, value), e.g.,
                [('gt', 3), ('lt', 4)]. It is necessary that
                len(constraints) == model.num_outputs.
            punchout_radius: Positive value defining the desired minimum distance between points
            bounds: torch.tensor whose first row is the lower bounds and second row is the upper bounds
            num_samples: Number of samples for MC integration
        """
        super().__init__(model=model, objective=IdentityMCObjective(), **kwargs)
        assert len(constraints) == model.num_outputs
        assert all(direction in ("gt", "lt") for direction, _ in constraints)
        assert punchout_radius > 0
        self.constraints = constraints
        self.punchout_radius = punchout_radius
        self.bounds = bounds
        self.lambda_ = lambda_
        self.base_points = train_inputs
        self.num_samples = num_samples
        self.ball_of_points = self._generate_ball_of_points(
            num_samples=num_samples,
            radius=punchout_radius,
            device=bounds.device,
            dtype=bounds.dtype,
        )
        self._thresholds = torch.tensor(
            [threshold for _, threshold in self.constraints]
        ).to(bounds)
        assert (
            all(ub > lb for lb, ub in self.bounds.T) and len(self.bounds.T) == self.dim
        )

    @property
    def num_outputs(self):
        return self.model.num_outputs

    @property
    def dim(self):
        return self.base_points.shape[-1]

    @property
    def train_inputs(self):
        return self.model.models[0].train_inputs[0]

    def _generate_ball_of_points(
        self, num_samples, radius, device=None, dtype=torch.double
    ):
        """Creates a ball of points to be used for MC."""
        tkwargs = {"device": device, "dtype": dtype}
        z = sample_hypersphere(d=self.dim, n=num_samples, qmc=True, **tkwargs)
        r = torch.rand(num_samples, 1, **tkwargs) ** (1 / self.dim)
        return radius * r * z
    
    def _generate_ball_of_points_metric(
        self, num_samples, radius, model,device=None, dtype=torch.double
    ):
        """Creates a ball of points to be used for MC."""
        tkwargs = {"device": device, "dtype": dtype}
        z = sample_hypersphere(d=self.dim, n=num_samples, qmc=True, model=model,**tkwargs)
        r = torch.rand(num_samples, 1, **tkwargs) ** (1 / self.dim)
        return radius * r * z

    def _get_base_point_mask(self, X):
        distance_matrix = self.model.models[0].covar_module.covar_dist(
            X.float(), self.base_points.float()
        )
        # distance_matrix2 = self.model.models[1].covar_module.covar_dist(
        #     X.float(), self.base_points.float()
        # )
        # distance_matrix = torch.stack((distance_matrix1,distance_matrix2),dim=-1)
        # distance_matrix = cdist(X.squeeze().cpu().float(),self.base_points.cpu().float())
        # breakpoint()
        return smooth_mask(distance_matrix, self.punchout_radius)

    def _get_base_point_mask_metric(self,X):
        posterior_X = self.model.posterior(X.float()).mean
        posterior_base = self.model.posterior(self.base_points.float()).mean
        distance_matrix = cdist(posterior_X.detach().cpu().reshape(-1,posterior_X.shape[-1]),posterior_base.detach().cpu())
        return smooth_mask(torch.tensor(distance_matrix), self.punchout_radius).reshape(X.shape[0],X.shape[1],-1)
        
        # cdist(self.base_points.)
                                    
    # def _get_base_point_mask(self, X):
    #     cost_X = self.model.models[0].posterior(
    #         X.float())
    #     cost_base_points = self.model.models[0].posterior(
    #         self.base_points.float())
    #     cdist_ = cdist(cost_X.cpu().detach().numpy(), cost_base_points.cpu().detach().numpy())
        
    #     breakpoint()
    #     return smooth_mask(distance_matrix, self.punchout_radius)

    def _estimate_probabilities_of_satisfaction_at_points(self, points):
        """Estimate the probability of satisfying the given constraints."""
        posterior = self.model.posterior(X=points.float())
        mus, sigma2s = posterior.mean, posterior.variance
        dist = torch.distributions.normal.Normal(mus, sigma2s.sqrt())
        norm_cdf = dist.cdf(self._thresholds)
        probs = torch.ones(points.shape[:-1]).to(points)
        for i, (direction, _) in enumerate(self.constraints):
            probs = probs * (
                norm_cdf[..., i] if direction == "lt" else 1 - norm_cdf[..., i]
            )
        return probs

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """Evaluate Expected Improvement on the candidate set X."""
        # lambda_ = 1.0
        ball_around_X = self.ball_of_points + X
        domain_mask = smooth_box_mask(
            ball_around_X, self.bounds[0, :], self.bounds[1, :]
        ).prod(dim=-1)
        num_points_in_integral = domain_mask.sum(dim=-1)
        base_point_mask2 = self._get_base_point_mask_metric(ball_around_X).prod(dim=-1)#.to(device='cuda')
        base_point_mask1 = self._get_base_point_mask(ball_around_X).prod(dim=-1)
        prob = self._estimate_probabilities_of_satisfaction_at_points(ball_around_X)
        # masked_prob = prob*domain_mask*base_point_mask2
        # masked_prob = prob*domain_mask*base_point_mask1
        masked_prob = prob * domain_mask * ((1-self.lambda_)*base_point_mask2+self.lambda_*base_point_mask1)
        y = masked_prob.sum(dim=-1) / num_points_in_integral
        return y

def get_and_fit_gp(X, Y):
    """Simple method for creating a GP with one output dimension.

    X is assumed to be in [0, 1]^d.
    """
    # assert Y.ndim == 2 and Y.shape[-1] == 1
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-6, 1e-3))  # Noise-free
    octf = Standardize(m=1)
    gp = SingleTaskGP(X, Y, likelihood=likelihood,outcome_transform=octf)
    mll = ExactMarginalLogLikelihood(model=gp, likelihood=gp.likelihood)
    fit_gpytorch_mll(mll)
    return gp