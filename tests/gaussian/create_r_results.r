# some simple code to produce a table of coefficients
# that we'll use as references to make sure the python
# version is working appropriately

data <- read.csv('gaussian_data_basic.csv')


model <- glm(y ~ x_00 +
             x_01 +
             x_02 +
             x_03, data=data, family=gaussian(link='identity'))

write.csv(data.frame(model$coefficients), 'gaussian_data_basic_coeff_r.csv')


model <- glm(y ~ x_00 +
             x_01 +
             x_02 +
             x_03, data=data, family=gaussian(link='log'))

write.csv(data.frame(model$coefficients), 'gaussian_data_basic_coeff_r_log.csv')


model <- glm(y ~ x_00 +
             x_01 +
             x_02 +
             x_03, data=data, family=gaussian(link='inverse'))

write.csv(data.frame(model$coefficients), 'gaussian_data_basic_coeff_r_inverse.csv')
