library(tidyverse)
library(jsonlite)
library(tidytext)
library(tidymodels)
library(vroom)
library(textrecipes)

## Initial Work
#####
# 
# ### Read in Training Data
# 
# trainData <- read_file('train.json') %>%
#   fromJSON() %>%
#   mutate(cuisine = as.factor(cuisine))
# 
# ### Read in Test Data
# 
# testData <- read_file('test.json') %>%
#   fromJSON()
# 
# ### Potential Features - Ingredient Count, Salt or Not, Baking
# 
# trainData <- trainData %>%
#   mutate(salted = as.integer(map_lgl(ingredients, ~ "salt" %in% .x))) %>%
#   mutate(baking = as.integer(map_lgl(ingredients, ~ 
#                                        any(tolower(.x) %in% 
#                                              c("baking soda", 
#                                                "baking powder"))))) %>%
#   mutate(count = map_int(ingredients, length)) %>%
#   mutate(cuisine = as.factor(cuisine))
# 
# testData <- testData %>%
#   mutate(salted = as.integer(map_lgl(ingredients, ~ "salt" %in% .x))) %>%
#   mutate(baking = as.integer(map_lgl(ingredients, ~ 
#                                        any(tolower(.x) %in% 
#                                              c("baking soda", 
#                                                "baking powder"))))) %>%
#   mutate(count = map_int(ingredients, length))
# 
# cook_recipe <- recipe(cuisine ~ ., data = trainData) %>%
#   step_rm(id, ingredients) %>%
#   step_mutate(salted = factor(salted)) %>%
#   step_mutate(baking = factor(baking))
# 
# cook_prep <- prep(cook_recipe)
# bake(cook_prep, new_data = trainData)
# 
# ### Trying Random Forest
# 
# library(rpart)
# 
# tree_mod <- rand_forest(mtry=tune(),
#                         min_n=tune(),
#                         trees=100) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# tree_workflow <- workflow() %>%
#   add_recipe(cook_recipe) %>%
#   add_model(tree_mod)
# 
# #### Grid of values to tune Over
# 
# tuning_grid <- grid_regular(mtry(range=c(1,3)),
#                             min_n(),
#                             levels=5)
# 
# #### CV
# 
# folds <- vfold_cv(trainData, v = 5, repeats = 1)
# 
# CV_results <- tree_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=(metric_set(roc_auc)))
# 
# #### Find best tuning parameters
# 
# bestTune <- CV_results %>%
#   select_best(metric="roc_auc")
# 
# #### Finalize workflow
# 
# final_wf <-
#   tree_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=trainData)
# 
# #### Predict
# 
# tree_predictions <- final_wf %>%
#   predict(new_data = testData, type="class")
# 
# #### Kaggle
# 
# tree_kaggle_submission <- tree_predictions %>%
#   bind_cols(., testData) %>%
#   select(id, .pred_class) %>%
#   rename(cuisine=.pred_class)
# 
# vroom_write(x=tree_kaggle_submission, file="./RandForPreds.csv", delim=',')
# 
#####

## Trying TF-IDF
#####

### Read in Data

trainData <- read_file("train.json") %>%
  fromJSON() %>%
  step_mutate(cuisine = factor(cuisine))

testData <- read_file("test.json") %>%
  fromJSON()

### Recipe

# tfidf_recipe <- recipe(cuisine ~ ingredients, data = trainData) %>%
#   step_mutate(ingredients = tokenlist(ingredients)) %>%
#   step_tokenfilter(ingredients, max_tokens=500) %>%
#   step_tfidf(ingredients)
# 
# tfidf_prep <- prep(tfidf_recipe)
# bake(tfidf_prep, new_data = trainData)

### New Recipe

new_recipe <- recipe(cuisine ~ ingredients, data = trainData) %>%
  step_mutate(ingredients = tokenlist(ingredients)) %>%
  step_tokenfilter(ingredients, max_tokens=1500) %>%
  step_tfidf(ingredients)

new_prep <- prep(new_recipe)
bake(new_prep, new_data = trainData)

#####

## Random Forest - Score: 0.68211, More Trees: 0.68423
#####

# library(rpart)
# 
# tree_mod <- rand_forest(mtry=tune(),
#                         min_n=tune(),
#                         trees=100) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# tree_mod <- rand_forest(mtry=tune(),
#                         min_n=tune(),
#                         trees=500) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# tree_workflow <- workflow() %>%
#   add_recipe(tfidf_recipe) %>%
#   add_model(tree_mod)
# 
# tree_workflow <- workflow() %>%
#   add_recipe(new_recipe) %>%
#   add_model(tree_mod)
# 
# ### Grid of values to tune Over
#
# tuning_grid <- grid_regular(mtry(range = c(1,20)),
#                             min_n(),
#                             levels=5)
# 
# tuning_grid <- grid_regular(mtry(range = c(20, 80)),
#                             min_n(range = c(5, 50)),
#                             levels = 5)
# 
# ### CV
# 
# folds <- vfold_cv(trainData, v = 5, repeats = 1)
# 
# CV_results <- tree_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=(metric_set(accuracy)))
# 
# ### Find best tuning parameters
# 
# bestTune <- CV_results %>%
#   select_best(metric="accuracy")
# 
# ### Finalize workflow
# 
# final_wf <-
#   tree_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=trainData)
# 
# ### Predict
# 
# tree_predictions <- final_wf %>%
#   predict(new_data = testData, type="class")
# 
# ### Kaggle
# 
# tree_kaggle_submission <- tree_predictions %>%
#   bind_cols(., testData) %>%
#   select(id, .pred_class) %>%
#   rename(cuisine=.pred_class)
# 
# vroom_write(x=tree_kaggle_submission, file="./RF_TDIDF_Preds.csv", delim=',')
#
# vroom_write(x=tree_kaggle_submission, file="./RF_MT_Preds.csv", delim=',')

#####

## SVM - Score:
#####

library(LiblineaR)

svmLinear <- svm_linear(mode = "classification") %>%
  set_engine("LiblineaR")

linear_workflow <- workflow() %>%
  add_recipe(new_recipe) %>%
  add_model(svmLinear) %>%
  fit(data=trainData)

#### Predict

linear_predictions <- predict(linear_workflow, new_data=testData, type="class")

#### Kaggle

linear_kaggle_submission <- linear_predictions %>%
  bind_cols(., testData) %>%
  select(id, .pred_class) %>%
  rename(cuisine=.pred_class)

vroom_write(x=linear_kaggle_submission, file="./LinearSVMPreds.csv", delim=',')

#####