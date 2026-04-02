import abc

from typing import TypeVar, Generic

T = TypeVar('T')

class CostFunctionInterface(Generic[T], abc.ABC):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, 'preiteration') and
            callable(subclass, subclass.preiteration) and
            hasattr(subclass, 'cost_function') and
            callable(subclass, subclass.cost_function) and
            hasattr(subclass, 'cost_selector') and
            callable(subclass, subclass.cost_selector) or
            NotImplement
        )

    @abc.abstractmethod
    def preiteration(
        self,
        Hr_terms,
        basis,
        overlap
    ):
        """
        Method to run prior to each iteration.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cost_function(
        self,
        Hr_terms,
        basis,
        overlap,
        training_grid,
        grid_index
    ) -> T:
        """
        Returns the cost of a training point
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cost_selector(
        self,
        costs: dict[int, T]
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
        raise NotImplementedError

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
