df <- read.csv("Cancer_Data.csv")
dim(df)
head(df,4) ## check the first 4 lines


## Following procedure is to manage the database to what we want, and divide different cases for benign and malignant tumors.
df <- df[,-c(1,33)]
X <- df[,-1]
Y <- df[,1]

X <- apply(X, 2, function(x) (x-min(x)) / (max(x)-min(x))) # Standardize all data into the range (0,1), inclusive.
Y[Y == 'M'] <- 1
Y[Y == 'B'] <- 0
Y <- as.factor(Y)

d_new <- as.data.frame(cbind(X, Y))
d_new$Y <- as.factor(d_new$Y - 1)
head(d_new,3)

# ensure results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)
# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model, with the method of LVQ learning (Learning Vectors Quantization)
model <- train(Y~., data=d_new, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)

## Another option is to train the model with Recursive Feature Elimination (RFE) method.
## load the data
data(PimaIndiansDiabetes)
# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(d_new[,1:30], d_new[,31], sizes=c(1:30), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))


d_new <- as.data.frame(cbind(X, Y))
d_new$Y <- as.factor(d_new$Y - 1)
head(d_new,3)

d <- importance[1]$importance
names <- rownames(d[d[,1] > 0.8,])
names <- c(names,'Y')

# Select the data with only 'importance' and output
d_new <- d_new[,colnames(d_new) %in% names]
head(d_new,3)
dim(d_new)

library(caTools)
library(ROCR)

split <- sample.split(d_new, SplitRatio = 0.8) ## Split the model into training set and practice set.
split

train_reg <- subset(d_new, split == "TRUE")
test_reg <- subset(d_new, split == "FALSE")
head(train_reg,5)

# Training model
library(car)

logistic_model <- glm(Y ~ .,
                      data = train_reg,
                      family = binomial(link='logit'))
logistic_model


predict_reg <- predict(logistic_model,
                       test_reg, type = "response")
predict_reg[1:7]
predict_reg <- ifelse(predict_reg >0.5, 1, 0)
table(test_reg$Y, predict_reg)
missing_classerr <- mean(predict_reg != test_reg$Y)
print(paste('Accuracy =', 1 - missing_classerr))
ROCPred <- prediction(predict_reg, test_reg$Y)
ROCPer <- performance(ROCPred, measure = "tpr",
                      x.measure = "fpr")

# Use AUC to reflect the accuracy of our model
auc <- performance(ROCPred, measure = "auc")
auc <- auc@y.values[[1]]
auc

plot(ROCPer)
plot(ROCPer, colorize = TRUE,
     print.cutoffs.at = seq(0.1, by = 0.1),
     main = "ROC CURVE")
abline(a = 0, b = 1)
# Compute the AUC value
auc <- round(auc, 4)
legend(.6, .4, auc, title = "AUC", cex = 1, pch=0)

# This is the end of the data part. This data analysis corresponds closely to the paper Prediction of Infecting Cancer-Based on Logistic Regression Model (LUO,2024)
