import numpy as np
import scipy as sp

from dataclasses import dataclass
from gaussian import *
from surrogate import SurrogateModel

@dataclass
class EvaluatedGEPPoint:
    theta: tuple
    energy: float
    ground_state: list[float]

class GEPAdvisor:
    energy_gp: GaussianProcess
    ground_state_gps: list[GaussianProcess]

    def __init__(
        self,
        model: SurrogateModel,
        param_bounds: list[tuple[float, float]],
        gp_class: type[GaussianProcess] = SklearnGP,
        seed: int | None = None
    ):
        self.model = model
        self.param_bounds = param_bounds
        self.energy_gp = gp_class()
        self.ground_state_gps = [gp_class() for _ in range(self.model.size)]

        self.evaluations = []
        self.rng = np.random.default_rng(seed)
        self.sobol = sp.stats.qmc.Sobol(len(param_bounds), rng = self.rng)

    def sobol_sample(self):
        samples = self.sobol.random_base2(2)
        self.samples = []
        thetas = []
        energies = []

        # add each point to the GP
        for sample in samples:
            theta = [
                (b[1] - b[0]) * s + b[0]
                for b, s in zip(self.param_bounds, sample)
            ]

            self.samples.append(theta)
            training_point = self.model.theta_to_training_point(theta)
            evals, evecs = self.model.solve(training_point)

            print(f"Sampled {theta} with energy {evals[0]}")
            thetas.append(theta)
            energies.append(evals[0])

        self.record_evaluations(thetas, energies)

    def record_evaluations(
        self,
        thetas: list[tuple],
        energies: list[float],
    ):
        for t, e in zip(thetas, energies):
            self.evaluations.append(
                EvaluatedGEPPoint(
                    theta=t,
                    energy=e,
                    ground_state=0
                )
            )

        self.refit()

    def refit(self):
        X = np.array([e.theta for e in self.evaluations])
        y_energy = np.array([e.energy for e in self.evaluations])
        y_ground_state = np.array([e.ground_state for e in self.evaluations]).T

        self.energy_gp.fit(X, y_energy)

        """
        for gp, y in zip(self.ground_state_gps, y_ground_state):
            gp.fit(X, y)
        """

    def predict(
        self,
        theta
    ):
        X = np.array(theta)
        X = X.reshape(1, -1)
        energy_mean, energy_std = self.energy_gp.predict(X)
        """
        ground_state_mean = []

        for gp in self.ground_state_gps:
            mean, std = gp.predict(X)
            ground_state_mean.append(mean[0])
        """

        return energy_mean[0]
