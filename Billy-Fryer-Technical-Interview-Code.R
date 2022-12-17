# Library
library(tidyverse)
library(randomForest)
library(xgboost)

# Read in Data and make SpinRate numeric bc the NULL values make it read
# in as character
raw_training <- read_csv("training.csv") %>% 
  mutate(SpinRate = as.numeric(SpinRate),
         # Need a Row Number variable for sampling procedure
         RowNumber = row_number())

deploy_data <- read_csv("deploy.csv") %>% 
  mutate(SpinRate = as.numeric(SpinRate))

# Split into training and testing sets
# Since we have such a class imbalance from count(raw_training, InPlay)
# I'm going to stratify this sampling roughly 70/30
set.seed(34)

# Split data by InPlay
train_data <- raw_training %>% 
  group_by(InPlay) %>% 
  slice_sample(prop = 0.7) %>% 
  ungroup()

# Test data is everything that wasn't in training data
test_data <- raw_training[-train_data$RowNumber,] %>% 
  select(-RowNumber)

# Ramove RowNumber from train_data
train_data <- select(train_data, -RowNumber)


## ----logistic-regression---------------------------------------------------------
# First try had Spin Rate non significant, so I'm going to refit it 
# Without that variable
logistic_model <- glm(InPlay ~ Velo + HorzBreak + InducedVertBreak,
                      data = train_data,
                      family = "binomial")
# Quick Summary
summary(logistic_model)

# Prediction on the test test
predict.test <- predict(logistic_model,
                        test_data,
                        type = "response")

# Bind predictions to test_data
data.test <- bind_cols(test_data, predictions = predict.test)

# Performance checks
performance::model_performance(logistic_model) # Log Loss = 0.5804942
performance::performance_accuracy(logistic_model) # 56.36%... Not that good


## ----random-forest---------------------------------------------------------------
# Need InPlay to be a factor for random forst to work
rf.model <- randomForest(as.factor(InPlay) ~ ., 
                        data = train_data, 
                        ntree = 100,
                        importance = TRUE,
                        proximity = TRUE,
                        # Rough Fix of the NAs in the SpinRate Column
                        na.action = na.roughfix)

rf.model

# This is better than the logistic model but still not too great
# In particular, my missclassification rate of when the ball is hit in
# play is absolutely atrocious


## ----xgboost---------------------------------------------------------------------
# Separate data from label
xgboost.matrix <- train_data %>% select(-InPlay) %>% as.matrix()
xgboost.label <- train_data$InPlay

### Splitting Data into training and testing sets
n <- nrow(xgboost.matrix)
prop <- 0.6
set.seed(34)
train <- sample(n, size = n * prop)
xgboost.matrix.train <- xgboost.matrix[train,]
xgboost.matrix.test <- xgboost.matrix[-train,]
xgboost.label.train <- xgboost.label[train]
xgboost.label.test <- xgboost.label[-train]

# Train Model
xgboost.model <- xgboost(data = xgboost.matrix.train, 
                         label = xgboost.label.train, 
                         nrounds = 100,
                         objective = "binary:logistic")

# logloss: 0.284316 

# Predictions
xgb_preds <- predict(xgboost.model, xgboost.matrix.test) %>% 
  as.data.frame()

outcomes <- bind_cols(xgboost.label.test, xgb_preds) 
names(outcomes) <- c("actual", "expected_prob")
outcomes <- outcomes %>% 
  mutate(expected_class = case_when(expected_prob >= 0.5 ~ 1,
                                    TRUE ~ 0))
sum(outcomes$actual == outcomes$expected_class) / nrow(outcomes)
# Test Set Accuracy = 68.61%


# Feature Importance
importance <- xgb.importance(feature_names = colnames(xgboost.matrix.train),
                             model = xgboost.model)
head(importance)
xgb.plot.importance(importance_matrix = importance)

# Everything seems to have equal importance!
# I'm going to go with this model for the reduced logloss


## ----deploy-data-----------------------------------------------------------------
# Make Predictions for deploy data
deploy_data_pred <- predict(xgboost.model, as.matrix(deploy_data))

# Combine Deploy Data with Predictions
final_data <- bind_cols(deploy_data, Predictions = deploy_data_pred)

# Write to csv
write.csv(final_data,
          "Deploy-With-Predictions.csv")

