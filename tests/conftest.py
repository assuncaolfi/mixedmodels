from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sleepstudy() -> pd.DataFrame:
    return pd.read_csv(Path(__file__).parent / "data" / "sleepstudy.csv")
