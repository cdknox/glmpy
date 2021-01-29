#import glmpy
import glmpy
import pandas as pd

df = pd.DataFrame({
    'intercept': [1, 1, 1, 1],
    'x': [1, 2, 3, 4],
    'y': [1, 2, 3, 4],
})


glm = glmpy.model.GLM(family = glmpy.families.Poisson, link = glmpy.links.LogLink)
glm.fit(X = df[['x', 'intercept']].values, y = df['y'].values)
print(glm.params)

glm = glmpy.model.GLM(family = glmpy.families.Poisson, link = glmpy.links.IdentityLink)
glm.fit(X = df[['intercept', 'x']].values, y = df['y'].values)
print(glm.params)




