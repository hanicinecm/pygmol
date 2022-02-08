from typing import Union, Sequence, Dict

import pydantic


class PlasmaParameters(pydantic.BaseModel):

    radius: float
    length: float
    pressure: float
    power: Union[float, Sequence[float]]
    t_power: Sequence[float] = None
    feeds: Dict[str, float]
    t_end: float
