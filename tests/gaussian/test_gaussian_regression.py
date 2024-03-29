import numpy as np
import pandas as pd
import pytest

import glmpy.links
import glmpy.model


@pytest.mark.parametrize(
    "link_obj,link_suffix",
    [
        (glmpy.links.IdentityLink, ""),
        (glmpy.links.LogLink, "_log"),
        (glmpy.links.InverseLink, "_inverse"),
    ],
)
def test_simple_fit(link_obj, link_suffix):
    df = pd.read_csv("tests/gaussian/gaussian_data_basic.csv")
    df["intercept"] = 1

    link = link_obj
    glm = glmpy.model.GLM(family=glmpy.families.Gaussian, link=link)
    glm.fit(
        X=df[["intercept", "x_00", "x_01", "x_02", "x_03"]].values, y=df["y"].values
    )
    python_params = pd.DataFrame(
        {
            "names": [
                "(Intercept)",
                "x_00",
                "x_01",
                "x_02",
                "x_03",
            ],
            "values": glm.params,
        }
    )
    r_params = pd.read_csv(
        f"tests/gaussian/gaussian_data_basic_coeff_r{link_suffix}.csv"
    )
    r_params.columns = ["names", "values"]

    assert np.allclose(
        r_params.set_index("names")["values"]
        - python_params.set_index("names")["values"],
        0,
    )
