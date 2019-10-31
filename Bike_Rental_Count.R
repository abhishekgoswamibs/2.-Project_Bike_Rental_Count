# Bike Rental Count

setwd('D:/Data Science Edwisor/6. Projects/2. Project Two Bike Rental Count')
# loading the dataset.
dataset = read.csv('day.csv', na.strings = c(" ","", "NA"))

str(dataset)

# Extracting Meaningful Information from dtedaay.
library(lubridate)
dataset$dteday = ymd(dataset$dteday)
str(dataset)
# Extracting date from dteday
dataset$date = day(dataset$dteday)

# Missing Value Analysis
sum(is.na(dataset))
# No missing values found

# Outlier Analysis for numeric variable
cnames = c('temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'date')
library(ggplot2)
for (i in 1:length(cnames)) {
  assign(paste0("gn" , i), ggplot(aes_string(y = (cnames[i]), x = 'cnt'), 
                                  data = dataset)+
           stat_boxplot(geom = "errorbar", width = 0.5)+
           geom_boxplot(outlier.colour = "red", fill = "grey", outlier.shape = 18, outlier.size = 1,
                        notch = FALSE)+
           theme(legend.position = "bottom")+
           labs(y = cnames[i], x = "cnt")+
           ggtitle(paste("Box plot of responded for", cnames[i])))
}
gridExtra:: grid.arrange(gn1,gn2,gn3, ncol=3) 
gridExtra:: grid.arrange(gn4,gn5, ncol = 2)
gridExtra:: grid.arrange(gn6,gn7, ncol = 2)

# As we see variables hum, temp, atemp have outliers in them so we try to remove them.
for (i in cnames) {
  print(i)
  val = dataset[,i][dataset[,i] %in% boxplot.stats(dataset[,i])$out]
  dataset[,i][dataset[,i] %in% val] = NA
}
sum(is.na(dataset))
apply(dataset, 2, function(x) {sum(is.na(x))})
# # Finding the best method to impute the missing values
# dataset$casual[7] # Original value = 148
# dataset$casual[7] = NA
# # Mean
# dataset$casual[is.na(dataset$casual)] = mean(dataset$casual, na.rm = T)
# # Value obtained in mean = 732.8615
# # Median
# dataset$casual[is.na(dataset$casual)] = median(dataset$casual, na.rm = T)
# # Value obtained in median = 675

# Finalising Median for imputations
# 1. hum
dataset$hum = ifelse(is.na(dataset$hum),
                     ave(dataset$hum, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$hum)
# 2. Windspeed
dataset$windspeed = ifelse(is.na(dataset$windspeed),
                     ave(dataset$windspeed, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$windspeed)
# 3. casual
dataset$casual = ifelse(is.na(dataset$casual),
                     ave(dataset$casual, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$casual)
# Recheck for outliers
cnames = c('hum', 'windspeed', 'casual')
for (i in 1:length(cnames)) {
  assign(paste0("gn" , i), ggplot(aes_string(y = (cnames[i]), x = 'cnt'), 
                                  data = dataset)+
           stat_boxplot(geom = "errorbar", width = 0.5)+
           geom_boxplot(outlier.colour = "red", fill = "grey", outlier.shape = 18, outlier.size = 1,
                        notch = FALSE)+
           theme(legend.position = "bottom")+
           labs(y = cnames[i], x = "cnt")+
           ggtitle(paste("Box plot of responded for", cnames[i])))
}
gridExtra:: grid.arrange(gn1,gn2,gn3, ncol=3) 
# Still some outliers left in windspeed and cnt hence we repeat the steps again
for (i in cnames) {
  print(i)
  val = dataset[,i][dataset[,i] %in% boxplot.stats(dataset[,i])$out]
  dataset[,i][dataset[,i] %in% val] = NA
}
sum(is.na(dataset))
apply(dataset, 2, function(x) {sum(is.na(x))})
# Replacing NAs
# 1. Windspeed
dataset$windspeed = ifelse(is.na(dataset$windspeed),
                           ave(dataset$windspeed, FUN = function(x) mean(x, na.rm = TRUE)),
                           dataset$windspeed)
# 2. casual
dataset$casual = ifelse(is.na(dataset$casual),
                        ave(dataset$casual, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$casual)
# Recheck
cnames = c('windspeed', 'casual')
for (i in 1:length(cnames)) {
  assign(paste0("gn" , i), ggplot(aes_string(y = (cnames[i]), x = 'cnt'), 
                                  data = dataset)+
           stat_boxplot(geom = "errorbar", width = 0.5)+
           geom_boxplot(outlier.colour = "red", fill = "grey", outlier.shape = 18, outlier.size = 1,
                        notch = FALSE)+
           theme(legend.position = "bottom")+
           labs(y = cnames[i], x = "cnt")+
           ggtitle(paste("Box plot of responded for", cnames[i])))
}
gridExtra:: grid.arrange(gn1,gn2, ncol=3) 
# Still some outliers for casual again repeating the above steps
for (i in cnames) {
  print(i)
  val = dataset[,i][dataset[,i] %in% boxplot.stats(dataset[,i])$out]
  dataset[,i][dataset[,i] %in% val] = NA
}
sum(is.na(dataset))
apply(dataset, 2, function(x) {sum(is.na(x))})
# 1. casual
dataset$casual = ifelse(is.na(dataset$casual),
                        ave(dataset$casual, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$casual)
# Recheck
# Recheck
cnames = c('casual')
for (i in 1:length(cnames)) {
  assign(paste0("gn" , i), ggplot(aes_string(y = (cnames[i]), x = 'cnt'), 
                                  data = dataset)+
           stat_boxplot(geom = "errorbar", width = 0.5)+
           geom_boxplot(outlier.colour = "red", fill = "grey", outlier.shape = 18, outlier.size = 1,
                        notch = FALSE)+
           theme(legend.position = "bottom")+
           labs(y = cnames[i], x = "cnt")+
           ggtitle(paste("Box plot of responded for", cnames[i])))
}
gridExtra:: grid.arrange(gn1, ncol=3) 

# Feature Selection

# Correlation Analysis(for numeric variables)
library(corrgram)
corrgram(dataset[,c(10,11,12,13,14,15,17)], order = F,
         upper.panel = panel.pie, text.panel = panel.txt, main = "correlation plot")
# As we can see variables temp and atemp are highly correlated

# Chi-square test for categorical variables.
# 1. season
cat_names = c('season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit')
for (i in cat_names) {
  print(i) 
  print(chisq.test(table(dataset$season, dataset[,i])))
}
# variables weathersit, mnth and season are dependent variables.
# 2. yr
for (i in cat_names) {
  print(i) 
  print(chisq.test(table(dataset$yr, dataset[,i])))
}
# 3. mnth
for (i in cat_names) {
  print(i) 
  print(chisq.test(table(dataset$mnth, dataset[,i])))
}
# 4. holiday
for (i in cat_names) {
  print(i) 
  print(chisq.test(table(dataset$holiday, dataset[,i])))
}
# we see that variables holiday, weekday, workingday are dependent on one another
# 5. weekday
for (i in cat_names) {
  print(i) 
  print(chisq.test(table(dataset$weekday, dataset[,i])))
}
# 6. workingday
for (i in cat_names) {
  print(i) 
  print(chisq.test(table(dataset$workingday, dataset[,i])))
}
# 7. weathersit
for (i in cat_names) {
  print(i) 
  print(chisq.test(table(dataset$weathersit, dataset[,i])))
}
# from above we see that variable temp and atemp are correlated, variables
# season, month and weathersit are dependent to eachother and variables holiday,
# weekday and workingday are also dependent on eachother.
# We keep temp, weathersit and holiday and remove rest of the dependent variables from our
# dataset.

# removing not required variables from our dataset
dataset = dataset[, c(4,6,9,10,12,13,14,15,16,17)]

# visualizations
# 1. yr
ggplot() + 
  geom_point(aes(x = dataset$yr, y = dataset$cnt),
             colour = 'red') +
  ggtitle('Bike_Count Vs Year') +
  xlab('Year') +
  ylab('Bike_Count')

# 2. holiday
ggplot() + 
  geom_point(aes(x = dataset$holiday, y = dataset$cnt),
             colour = 'red') +
  ggtitle('Bike_Count Vs Holiday') +
  xlab('Holiday') +
  ylab('Bike_Count')

# 3. weathersit
ggplot() + 
  geom_point(aes(x = dataset$weathersit, y = dataset$cnt),
             colour = 'red') +
  ggtitle('Bike_Count Vs Weather Situation') +
  xlab('weathersit') +
  ylab('Bike_Count')

# 4. temp
ggplot() + 
  geom_point(aes(x = dataset$temp, y = dataset$cnt),
             colour = 'red') +
  ggtitle('Bike_Count Vs Temperature') +
  xlab('Temperature') +
  ylab('Bike_Count')

# 5. hum
ggplot() + 
  geom_point(aes(x = dataset$hum, y = dataset$cnt),
             colour = 'red') +
  ggtitle('Bike_Count Vs Humidity') +
  xlab('Humidity') +
  ylab('Bike_Count')

# 6. windspeed
ggplot() + 
  geom_point(aes(x = dataset$windspeed, y = dataset$cnt),
             colour = 'red') +
  ggtitle('Bike_Count Vs Windspeed') +
  xlab('Windspeed') +
  ylab('Bike_Count')

# 7. casual
ggplot() + 
  geom_point(aes(x = dataset$casual, y = dataset$cnt),
             colour = 'red') +
  ggtitle('Bike_Count Vs Casual') +
  xlab('Casual') +
  ylab('Bike_Count')

# 8. Registered
ggplot() + 
  geom_point(aes(x = dataset$registered, y = dataset$cnt),
             colour = 'red') +
  ggtitle('Bike_Count Vs Registered') +
  xlab('Registered') +
  ylab('Bike_Count')

# Checking Distribution
# 1. yr
plot(density(dataset$yr))

# 2. holiday
plot(density(dataset$holiday))

# 3. weathersit
plot(density(dataset$weathersit))

# 4. temp
plot(density(dataset$temp))

# 5. hum
plot(density(dataset$hum))

# 6. windspeed
plot(density(dataset$windspeed))

# 7. casual
plot(density(dataset$casual))

# 8. Registered
plot(density(dataset$registered))
# No significant skewness found in our predictors

# Converting to proper datatypes
dataset$yr = as.factor(dataset$yr)
dataset$holiday = as.factor(dataset$holiday)
dataset$weathersit = as.factor(dataset$weathersit)
str(dataset)

# Splitting the data into train and test
library(caTools)

set.seed(123)
split = sample.split(dataset$cnt, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Building Models

# 1. Multiple Linear Regression Model
regressor_LR = lm(formula = cnt ~ .,
                  data = training_set)
summary(regressor_LR)

# Building Optimal model using backward elimination

# 1. Multiple Linear Regression Model
regressor_LR = lm(formula = cnt ~ yr + holiday + weathersit + temp + hum + windspeed + casual + registered + date,
                  data = training_set)
summary(regressor_LR)
# hum has pvlue > 0.05 hence we remove this variable as this carries not much information
# to explain our target bariable.
regressor_LR = lm(formula = cnt ~ yr + holiday + weathersit + temp + windspeed + casual + registered + date,
                  data = training_set)
summary(regressor_LR)
# Removing date now.
regressor_LR = lm(formula = cnt ~ yr + holiday + weathersit + temp + windspeed + casual + registered,
                  data = training_set)
summary(regressor_LR)
# # removing holiday
# regressor_LR = lm(formula = cnt ~ yr + weathersit + temp + windspeed + casual + registered,
#                   data = training_set)
# summary(regressor_LR)
# Adjusted R2 has decreased on removing holiday variable entirely hence we keep holiday
# in our dataset.

# # Removing windspeed 
# regressor_LR = lm(formula = cnt ~ yr + holiday + weathersit + temp + casual + registered,
#                   data = training_set)
# summary(regressor_LR)
# Adjusted R2 decreased on removing windspeed hence we keep windspeed as well.

# So we have our optimal model now

# Predicting the test set
y_pred = predict(regressor_LR, newdata = test_set)

# # Storing back the Inputs & Outputs
# write.csv(training_set, 'Sample_Input_R.csv', row.names = FALSE)
# backup = data.frame(Bike_Counts = y_pred)
# write.csv(backup, 'Sample_Output_R.csv', row.names = FALSE)

# MAPE CHECk
MAPE = function(y, x){
  mean(abs((y - x)/y)) * 100
}

MAPE(test_set[, 9], y_pred)

# Alternate method
library(DMwR)
regr.eval(test_set[,9], y_pred, stats = c('rmse', 'mape'))
# The performance of our Linear Model is good.

# visualizing predictions and actual values
ggplot() + 
  geom_point(aes(x = 1:nrow(test_set), y = test_set$cnt),
             colour = 'blue') +
  geom_point(aes(x = 1:nrow(test_set), y = y_pred),
             colour = 'red') +
  ggtitle('Predicted Vs Actual') +
  xlab('Observations') +
  ylab('Bike_Count')

# plotting a histogram of our residuals
diff = test_set$cnt - y_pred
plot(hist(x = diff, breaks = 20))

# Decision Tree Regressor
library(rpart)
regressor_DT = rpart(formula = cnt  ~ .,
                     data = training_set,
                     control = rpart.control(minsplit = 2))
summary(regressor_DT)

# Predicting the test set
y_pred_DT = predict(regressor_DT, newdata = test_set)

# Evaluating Performance
regr.eval(test_set[,9], y_pred_DT, stats = c('rmse', 'mape'))

# visualizing predictions and actual values
ggplot() + 
  geom_point(aes(x = 1:nrow(test_set), y = test_set$cnt),
             colour = 'blue') +
  geom_point(aes(x = 1:nrow(test_set), y = y_pred_DT),
             colour = 'red') +
  ggtitle('Predicted Vs Actual') +
  xlab('Observations') +
  ylab('Bike_Count')

# plotting a histogram of our residuals
diff = test_set$cnt - y_pred_DT
plot(hist(x = diff, breaks = 20))

# 3. Random Forest Model
# Fitting Random Forest Regression Model to the model
library(randomForest)
set.seed(1234)
regressor_RF = randomForest(x = training_set[, c(1,2,3,4,5,6,7,8,10)],
                            y = training_set$cnt,
                            ntree = 10)
                            
# Predicting the test set
y_pred_RF = predict(regressor_RF, newdata = test_set)
# Evaluating Performance
regr.eval(test_set[,9], y_pred_RF, stats = c('rmse', 'mape'))

# visualizing predictions and actual values
ggplot() + 
  geom_point(aes(x = 1:nrow(test_set), y = test_set$cnt),
             colour = 'blue') +
  geom_point(aes(x = 1:nrow(test_set), y = y_pred_RF),
             colour = 'red') +
  ggtitle('Predicted Vs Actual') +
  xlab('Observations') +
  ylab('Bike_Count')

# plotting a histogram of our residuals
diff = test_set$cnt - y_pred_RF
plot(hist(x = diff, breaks = 20))

# Random forest and linear regression model has comparable stats although random 
# forest is a very powerful model still linear regression model is a very good model for 
# regression problems.
# hence finalizing linear regression model for our project with best values of mape.

# Lets train the Linear model with the entire training set so that it could be used well
# with newer test sets.
regressor_LR_final = lm(formula = cnt ~ yr + holiday + weathersit + temp + windspeed + casual + registered,
                                       data = dataset)
summary(regressor_LR_final)

y_pred_final = predict(regressor_LR_final, newdata = dataset)

# Evaluating Performance(Train data)
regr.eval(dataset[,9], y_pred_final, stats = c('rmse', 'mape'))

# visualizing predictions and actual values
ggplot() + 
  geom_point(aes(x = 1:nrow(dataset), y = dataset$cnt),
             colour = 'blue') +
  geom_point(aes(x = 1:nrow(dataset), y = y_pred_final),
             colour = 'red') +
  ggtitle('Predicted Vs Actual(Training On Whole Dataset)') +
  xlab('Observations') +
  ylab('Bike_Count')

# plotting a histogram of our residuals
diff = dataset$cnt - y_pred_final
plot(hist(x = diff, breaks = 20))

################################## END ##################################################
