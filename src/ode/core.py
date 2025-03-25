from typing import Protocol, List
from numpy.typing import ArrayLike
from scipy.integrate import odeint
import pandas as pd


class ODEModel(Protocol):
    def __call__(self, t: float, state: ArrayLike) -> ArrayLike:
        """Calling the ODEModel for a given time `t` should return the instantaneous rate of change."""
        pass

    def names(self) -> List[str]:
        """The name of each state space variable."""
        pass


def solve_ode(
    time_points: ArrayLike, model: ODEModel, initial_conditions: ArrayLike
) -> pd.DataFrame:
    solution = odeint(model, initial_conditions, time_points)
    d = {name: solution[:, i] for i, name in enumerate(model.names())}
    d["time"] = time_points
    return pd.DataFrame(d)
