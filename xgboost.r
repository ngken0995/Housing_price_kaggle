library(dplyr)
library(mice)
library(caret)
library(data.table)
library(xgboost)
library(Metrics)



#set directory

setwd("~/data/housingprice")


#Read dataframe "trainhome.csv" and "testhome.csv"

trainhome <- read.csv("trainhome.csv")

testhome <- read.csv("testhome.csv")


#add a new column call "SalePrice" in testhome

testhome$SalePrice <- NA


#combine both data

data_combined <- rbind(trainhome, testhome)



#convert factor into integer

features=names(trainhome)


for(f in features){

  if(class(data_combined[[f]])=="factor"){

     levels=sort(unique(data_combined[[f]]))

       data_combined[[f]]=as.integer(factor(data_combined[[f]],levels = levels))

    }

}


#Find missing value and percantage by each column

propmiss <- function(dataframe) {

      m <- sapply(dataframe, function(x) {
            data.frame(
                  nmiss=sum(is.na(x)), 
                  n=length(x), 
                  propmiss=sum(is.na(x))/length(x)
            )
      })

      d <- data.frame(t(m))
      d <- sapply(d, unlist)
      d <- as.data.frame(d)
      d$variable <- row.names(d)
      row.names(d) <- NULL
      d <- cbind(d[ncol(d)],d[-ncol(d)])
      return(d[order(d$propmiss), ])

}


propmiss(data_combined)



drops <- c("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu")

data_combined <- data_combined[,!(names(data_combined) %in% drops)]

#sum "Fullbath" and "BsmtFullBath"

data_combined$TotalFullBath <- rowSums(data_combined[,c("FullBath", "BsmtFullBath")])

#sum "HalfBath" and "BsmtHalfBath"

data_combined$TotalHalfBath <- rowSums(data_combined[,c("HalfBath", "BsmtHalfBath")])

drops <- c("HalfBath", "BsmtHalfBath", "FullBath", "BsmtFullBath")

data_combined <- data_combined[,!(names(data_combined) %in% drops)]

#select all Near Zero variance and remove

nearfac <- nearZeroVar(data_combined)

nearfac

data_combined <- data_combined[,-nearfac]

#split x variable and y variable

x_data <- data_combined[,-51]

y_data <- data_combined[,51]

#remove id

x_data$Id <- NULL

#use mice package on x_data

x_implute <- complete(mice(x_data, m=2))

propmiss(x_data)

x_data$Exterior1st=x_implute$Exterior1st

x_data$Exterior2nd=x_implute$Exterior2nd

x_data$BsmtFinSF1=x_implute$BsmtFinSF1

x_data$BsmtUnfSF=x_implute$BsmtUnfSF

x_data$TotalBsmtSF=x_implute$TotalBsmtSF

x_data$Electrical = x_implute$Electrical

x_data$BsmtFinSF2=x_implute$BsmtFinSF2

x_data$KitchenQual=x_implute$KitchenQual

x_data$GarageCars=x_implute$GarageCars

x_data$GarageArea=x_implute$GarageArea

x_data$SaleType=x_implute$SaleType

x_data$TotalHalfBath=x_implute$TotalHalfBath

x_data$TotalFullBath=x_implute$TotalFullBath

x_data$MSZoning=x_implute$MSZoning

x_data$MasVnrArea=x_implute$MasVnrArea

x_data$MasVnrType=x_implute$MasVnrType

x_data$MSSubClass=x_implute$MSSubClass

x_data$BsmtFinType1=x_implute$BsmtFinType1

x_data$BsmtQual=x_implute$BsmtQual

x_data$BsmtExposure=x_implute$BsmtExposure

x_data$GarageFinish=x_implute$GarageFinish

x_data$GarageYrBlt=x_implute$GarageYrBlt

x_data$LotFrontage=x_implute$LotFrontage

x_data$GarageType=x_implute$GarageType

propmiss(x_data)

str(x_data)

train_update = as.data.table(x_data)

str(train_update)

#Separate train set X and test set X

HometrainX <- train_update[1:1460,]

HometestX <- train_update[1461:2919,]

#train set Y

HometrainY <- trainhome$SalePrice

#convert into numeric for XGBoost implementation

HometrainX[] <- lapply(HometrainX, as.numeric)

HometestX[]<-lapply(HometestX, as.numeric)

#start with train and test on train data.

set.seed(56)

Rows <- createDataPartition(HometrainY,
                            p = .80,
                            list= FALSE)

trainPredictors <- HometrainX[Rows,]
trainClasses <- HometrainY[Rows]

testPredictors <- HometrainX[-Rows,]
testClasses <- HometrainY[-Rows]

dtrain=xgb.DMatrix(as.matrix(trainPredictors),label= trainClasses)

dtest=xgb.DMatrix(as.matrix(testPredictors))

#xgboost parameters

xgb_params = list(

      seed = 0,

      colsample_bytree = 0.5,

      subsample = 0.8,

      eta = 0.02, 

      objective = 'reg:linear',

      max_depth = 12,

      alpha = 1,

      gamma = 2,

      min_child_weight = 1,

      base_score = 7.76

)


xg_eval_mae <- function (yhat, dtrain) {

      y = getinfo(dtrain, "label")

      err= mae(exp(y),exp(yhat) )

      return (list(metric = "error", value = err))

}


best_n_rounds=800 # try more rounds

#train data

gb_dt=xgb.train(xgb_params,dtrain,nrounds = as.integer(best_n_rounds))

train_submission=predict(gb_dt,dtest)

#root mean square log error 
rmse(log(testClasses),log(train_submission))

#model and submit real test
dtrain=xgb.DMatrix(as.matrix(HometrainX),label= HometrainY)

dtest=xgb.DMatrix(as.matrix(HometestX))

#test data

gb_dt=xgb.train(xgb_params,dtrain,nrounds = as.integer(best_n_rounds))

submission=fread('sample_submission.csv',colClasses = c("integer","numeric"))

submission$SalePrice=predict(gb_dt,dtest)

write.csv(submission,"xgb.csv",row.names = FALSE)
