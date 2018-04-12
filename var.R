setwd("C:/Users/s_zhangyw/Desktop/")
library(feather)
data <- read_feather("exrtn.ft")

library(vars)
model <- VAR(data, type="none", lag.max=12, ic="AIC")
summary(model)


fct <- predict(model, n.ahead=1, ci=0.90)
fct

VARselect(data, lag.max=24, type="none")
model <- VAR(data, p=1, type="const")


var.2c <- VAR(Canada, p = 2, type = "const")
predict(var.2c, n.ahead = 8, ci = 0.95)

sl <- VARselect(data, lag.max=12, type="const")
x <- sl[1]

def var(arr, lag):
  model = VAR(arr)
results = model.fit(maxlags=lag, ic='aic')
lag_order = results.k_ar
fct = results.forecast(arr[-lag_order:], 1)
return fct

def predict(df, lag=20, roll=250):
  idx = df.index[roll:]
clmn = df.columns
arr = df.values
brr = np.zeros((arr.shape[0]-roll, arr.shape[1]))
for i in range(roll, len(arr)):
  data = arr[i-roll:i, :]
brr[i-roll, :] = var(data, lag)
return pd.DataFrame(brr, columns=clmn, index=idx)

