import numpy as np
import pandas as pd

import glmpy.model


def test_simple_fit():
    # df = pd.read_csv('poisson_data_basic.csv')
    df = pd.read_csv("tests/poisson/poisson_data_basic.csv")
    df["intercept"] = 1

    link = glmpy.links.LogLink
    glm = glmpy.model.GLM(family=glmpy.families.Poisson, link=link)
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
    # r_params = pd.read_csv('poisson_data_basic_coeff_r.csv')
    r_params = pd.read_csv("tests/poisson/poisson_data_basic_coeff_r.csv")
    r_params.columns = ["names", "values"]

    print(python_params)
    print(r_params)
    # assert True
    # assert (python_params == r_params).all().all()
    assert np.allclose(
        r_params.set_index("names")["values"]
        - python_params.set_index("names")["values"],
        0,
    )
