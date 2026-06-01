import abc
import numpy as np

from typing import TypeVar, Generic, TYPE_CHECKING

T = TypeVar('T')

if TYPE_CHECKING:
    from surrogate2 import SurrogateModel

class CostFunctionInterface(Generic[T], abc.ABC):
    @abc.abstractmethod
    def preiteration(self):
        """
        Method to run prior to each iteration.
        """
        pass
    
    @abc.abstractmethod
    def gen_training_points(self) -> np.ndarray:
        """
        List of training points which are options to be used
        """
        pass

    @abc.abstractmethod
    def cost_function(
        self,
        training_point: dict
    ) -> T:
        """
        Returns the cost of a training point
        """
        pass

    @abc.abstractmethod
    def cost_selector(
        self,
        training_points: np.ndarray,
        costs: np.ndarray
    ) -> int:
        """
        Selects a training point from a dictionary with corresponding costs

        Paremeters
        ----------
        costs : dict[int, T]
            a dictionary of indices to costs

        Returns
        -------
        selected : `int`
            the index of the selected training point
        """
        pass

    @abc.abstractmethod
    def check_termination(
        self,
        iteration_costs: list[T]
    ) -> bool:
        """
        Whether or not to terminate

        Parameters
        ----------
        iteration_costs : `list[T]`
            a list of the chosen cost for each iteration

        Returns
        -------
        terminate : `bool`
            whether or not to terminate
        """
        pass

