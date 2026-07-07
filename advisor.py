import numpy as np
import scipy as sp

from costfunction import CostFunctionInterface
from dataclasses import dataclass
from gaussian import *
from pauli import *
from surrogate import SurrogateModel

@dataclass
class EvaluatedPoint:
    theta: np.ndarray
    cost: float

class SurrogateAdvisor:
    """
    We want to be able to calculate a extremum without making expensive
    calculation calls
    """

    model: SurrogateModel
    param_bounds: list[tuple[float, float]]
    exploration_weight: float
    diversity_weight: float
    log_sample_size: int

    cost_gp: GaussianProcess
    evaluations: list[EvaluatedPoint]
    rng: np.random.Generator
    samples: np.ndarray

    def __init__(
        self,
        model: SurrogateModel,
        cfi: CostFunctionInterface,
        param_bounds: list[tuple[float, float]],
        exploration_weight = 1.0,
        diversity_weight = 1.0,
        log_sample_size: int = 5,
        gp_class: type[GaussianProcess] = SklearnGP,
        seed: int | None = None
    ):
        self.model = model
        self.cfi = cfi
        self.param_bounds = param_bounds
        self.exploration_weight = exploration_weight
        self.diversity_weight = diversity_weight
        self.log_sample_size = log_sample_size

        self.cost_gp = gp_class()
        self.evaluations = []
        self.rng = np.random.default_rng(seed)
        self.sobol = sp.stats.qmc.Sobol(len(param_bounds), rng = self.rng)
        self.samples = None

    def sobol_sample(self):
        samples = self.sobol.random_base2(self.log_sample_size)
        self.samples = []
        thetas = []
        costs = []

        # add each point to the GP
        for sample in samples:
            theta = [
                (b[1] - b[0]) * s + b[0]
                for b, s in zip(self.param_bounds, sample)
            ]

            self.samples.append(theta)
            training_point = self.model.theta_to_training_point(theta)
            cost = self.cfi.cost_function(training_point)
            thetas.append(theta)
            costs.append(cost)

            print(f"Sampled {theta} with cost {cost}")

        self.record_evaluations(thetas, costs)

    def record_evaluations(
        self,
        thetas: list[np.ndarray],
        costs: list[float]
    ):
        for t, c in zip(thetas, costs):
            self.evaluations.append(
                EvaluatedPoint(
                    theta=t,
                    cost=c
                )
            )

        self.refit()

    def refit(self):
        X = np.array([e.theta for e in self.evaluations])
        y = np.array([e.cost for e in self.evaluations])

        self.cost_gp.fit(X, y)

    def predict(
        self,
        theta,
    ):
        X = np.array(theta)
        X = X.reshape(1, -1)
        mean, std = self.cost_gp.predict(X)

        return mean[0]

    def find_max(self) -> np.ndarray:
        result = sp.optimize.differential_evolution(
            lambda theta: -self.predict(theta),
            bounds=self.param_bounds,
            rng=self.rng
        )

        return result.x


################################################################################

    def propose_next_point(self) -> np.ndarray:
        if not self.cost_gp.is_fitted():
            return np.array([
                self.rng.uniform(lo, hi) for lo, hi in self.param_bounds
            ])

        result = sp.optimize.differential_evolution(
            self.acquisition,
            bounds=self.param_bounds,
            rng=self.rng
        )

        return result.x

    def diversity_score(
        self,
        theta: np.ndarray
    ):
        distances = [
            np.linalg.norm(
                theta - e.theta
                for e in self.evaluations
            )
        ]

        if distances == []:
            print("Check Required")
            return 1.0
        else:
            print("Check not required")

        return float(np.min(distances))

    def acquisition(
        self,
        theta: np.ndarray
    ) -> float:
        mean, std = self.cost_gp.predict(theta)
        cost_score = float(mean[0]) + self.exploration_weight * float(std[0])
        diversity_score = self.diversity_score(theta)

        score = cost_score * (1.0 + self.diveristy_weight * diversity)

        return -score
