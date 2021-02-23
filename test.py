import pandas as pd

import glmpy.model

df = pd.DataFrame(
    {
        "intercept": [1, 1, 1, 1],
        "x": [1, 2, 3, 4],
        "y": [1, 2, 3, 4],
    }
)

link = glmpy.links.LogLink
glm = glmpy.model.GLM(family=glmpy.families.Poisson, link=link)
glm.fit(X=df[["x", "intercept"]].values, y=df["y"].values)
print(glm.params)

link = glmpy.links.IdentityLink
glm = glmpy.model.GLM(family=glmpy.families.Poisson, link=link)
glm.fit(X=df[["intercept", "x"]].values, y=df["y"].values)
print(glm.params)
