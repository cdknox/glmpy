

#data <- data.frame(
#           x_00 = c(-1, 0, 1, 2, 3, 4),
#           x_01 = c(-6.5, 4, 2, 2, 1.4, 9),
#           x_02 = c(5, 1, 1, 1, 2.2, 0.5),
#           x_03 = c(0, 0, 1, 1, 1, 0),
#           y = c(1, 2, 1, 8, 3, 4)
#
#        )
data <- read.csv('poisson_data_basic.csv')


model <- glm(y ~ x_00 +
             x_01 +
             x_02 +
             x_03, data=data, family=poisson(link='log'))

write.csv(data.frame(model$coefficients), 'poisson_data_basic_coeff_r.csv')

print(model)
#print(model$coef)
#print(class(model$coef))
#print(rownames(model$coefficients))
#print(colnames(model$coefficients))
print(data.frame(model$coefficients))
#print(data.frame(variable = row.names(model$coef),
#                 coefficients = model$coef
#                 )
#)
