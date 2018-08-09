data <- read.csv("Rdata.csv", row.names="Date")
y <- lm(y_e~cpi+m2+rate+gdp+y_b, data)
y <- lm(corr~diff(cpi)+diff(m2)+diff(rate)+diff(gdp), data)

summary(y)

y <- lm(data$corr~data$cpi+data$m2)
summary(y)
r <- ts(y$residuals)

ecm <- lm(diff(data$corr)~diff(data$cpi)+diff(data$m2)+r[1:185])

y_p <- predict(ecm, diff(data$cpi)+diff(data$m2)+r[1:185]