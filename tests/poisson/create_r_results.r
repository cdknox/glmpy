# some simple code to produce a table of coefficients
# that we'll use as references to make sure the python
# version is working appropriately

data <- read.csv('poisson_data_basic.csv')


model <- glm(y ~ x_00 +
             x_01 +
             x_02 +
             x_03, data=data, family=poisson(link='log'))

write.csv(data.frame(model$coefficients), 'poisson_data_basic_coeff_r.csv')


model <- glm(y ~ x_00 +
             x_01 +
             x_02 +
             x_03, data=data, family=poisson(link='identity'))

write.csv(data.frame(model$coefficients), 'poisson_data_basic_coeff_r_identity.csv')


model <- glm(y ~ x_00 +
             x_01 +
             x_02 +
             x_03, data=data, family=poisson(link='sqrt'))

write.csv(data.frame(model$coefficients), 'poisson_data_basic_coeff_r_sqrt.csv')
