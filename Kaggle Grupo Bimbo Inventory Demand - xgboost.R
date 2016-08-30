#Solution to Kaggle competition: Grupo Bimbo Inventory Demand
#Author: Matt Brown     email: Matthew.brown.iowa@gmail.com
#
#Used xgboost package in R
#

library(data.table)
library(xgboost)
library(Matrix)
library(MatrixModels)
options(scipen = 999)

###Read in datasets###
train <- fread("C:/Users/Matt/Desktop/train.csv", header = TRUE, 
               select = c("Semana", "Producto_ID", "Cliente_ID", "Demanda_uni_equil"))
train <- train[Semana >6,]

##Load test dataset into a matrix
test<-fread("C:/Users/Matt/Desktop/test.csv", header = TRUE, 
            select = c("id", "Semana", "Producto_ID", "Cliente_ID", "Demanda_uni_equil"))

train$id <- 0
test$Demanda_uni_equil <-0

train$tst <- 0
test$tst <- 1

data <- rbind(train, test)
rm(test)
rm(train)


#Created two lag variables, L1 and L2, which were the Demanda_uni_equil from the 
#two previous weeks based on Cliente_ID and Producto_ID

data1<-data[,.(Semana=Semana+1,Producto_ID,Cliente_ID,Demanda_uni_equil)]
data<-merge(data,data1[Semana>8,.(L1=mean(Demanda_uni_equil)),
                       by =.(Semana,Producto_ID,Cliente_ID)],all.x=TRUE,
            by = c("Semana","Producto_ID","Cliente_ID"))
data1<-data[,.(Semana=Semana+2,Cliente_ID,Producto_ID,Demanda_uni_equil)]
data=merge(data,data1[Semana>8,.(L2=mean(Demanda_uni_equil)), by=.(Semana,Cliente_ID,Producto_ID)],all.x=T, by=c("Semana","Cliente_ID","Producto_ID"))
rm(data1)


#Count number of rows for the feature variables.
nProducto_ID=data[,.(nProducto_ID=.N),by=.(Producto_ID,Semana)]
nProducto_ID=nProducto_ID[,.(nProducto_ID=mean(nProducto_ID,na.rm=T)),by=Producto_ID]
data = merge(data,nProducto_ID, by = "Producto_ID", all.x = TRUE)
rm(nProducto_ID)

nCliente_ID=data[,.(nCliente_ID=.N),by=.(Cliente_ID,Semana)]
nCliente_ID=nCliente_ID[,.(nCliente_ID=mean(nCliente_ID,na.rm=T)),by=Cliente_ID]
data = merge(data,nCliente_ID, by = "Cliente_ID", all.x = TRUE)
rm(nCliente_ID)

data_train <- data[tst==0,]
data_test <- data[tst==1,]
data_test <-data_test[order(id),]
rm(data)
rm(data1)
gc()

#Subset dataset because it is too large to read into memory
data_train.sample <- data_train[sample(1:nrow(data_train), 10000000,replace=FALSE),]

#Create a subset which is used to create a lagged variable for week 11 predictions
data_train.subset<-subset(data_train, Semana>8, select = c("Semana","Producto_ID","Cliente_ID","L1",
                                                           "L2","Demanda_uni_equil"))
rm(data_train)

####Separate train set into predictor and response variables
data_train.sample.data <-  subset(data_train.sample, select = -Demanda_uni_equil)
data_train.sample.label <- subset(data_train.sample, select = Demanda_uni_equil)
rm(data_train.sample)

data_train.sample.data$L1[is.na(data_train.sample.data$L1)]<-0
data_train.sample.data$L2[is.na(data_train.sample.data$L2)]<-0

#Create sparse matrix
#constrasts.arg are the variables that will be treated as factors

data_train.matrix.data <- sparse.model.matrix(~id+Semana+nCliente_ID+nProducto_ID+
                                                +L1+L2+Producto_ID+Cliente_ID,
                                              data=data_train.sample.data,
                                              contrasts.arg = c("Producto_ID","Cliente_ID"),
                                              sparse = TRUE, sci = FALSE)
rm(data_train.sample.data)
data_train.matrix.label <- sapply(data.frame(data_train.sample.label),as.numeric)
rm(data_train.sample.label)

#Create matrix to use in xgboost function
data_train.DMatrix <-xgb.DMatrix(data = data_train.matrix.data,
                                 label = data_train.matrix.label)

rm(data_train.matrix.data)
rm(data_train.matrix.label)

#Cross validation
#Determines number of iterations to fun model
xgb.tab = xgb.cv(data = data_train.DMatrix, objective = "reg:linear", booster = "gbtree",
                 nrounds = 500, nfold = 4, early.stop.round = 10, 
                 evaluation = "rmse", nthreads = 10, eta = .1, max_depth = 10, maximize = FALSE)


min.error.idx = which.min(xgb.tab[, test.rmse.mean])

#Create model
model<- xgboost(data = data_train.DMatrix, objective = "reg:linear", booster = "gbtree",
                nrounds = min.error.idx, evaluation = "rmse", nthreads = 10, maximize = FALSE,
                eta = .1, max_depth = 10 )

data_test$L1[is.na(data_test$L1)]<-0
data_test$L2[is.na(data_test$L2)]<-0
data_test.matrix<-sparse.model.matrix(~id+Semana+nCliente_ID+nProducto_ID+
                                        +L1+L2+Producto_ID+Cliente_ID,
                                      data=data_test,
                                      contrasts.arg = c("Producto_ID","Cliente_ID"),
                                      sparse = TRUE, sci = FALSE)


data_test.matrix10<- data_test.matrix[data_test.matrix[,"Semana"]==10,]

# Predict week 10 test data using the trained model
pred1 <- predict(model, data_test.matrix10)  

pred1[pred1 < 0] <- 0
data_test[Semana==10]$Demanda_uni_equil<-pred1

#Predict week 11 using weeks 9 and 10 test data
data_test.11 <- subset(data_test, Semana==11)
data1<-data_test[,.(Semana=Semana+1,Producto_ID,Cliente_ID,Demanda_uni_equil)]
head(data1)
data_test.11<-merge(data_test.11,data1[Semana==11,.(L3=mean(Demanda_uni_equil)),
                                       by =.(Semana,Producto_ID,Cliente_ID)],all.x=TRUE,
                    by = c("Semana","Producto_ID","Cliente_ID"))
data1<-data_train.subset[,.(Semana=Semana+2,Producto_ID,Cliente_ID,Demanda_uni_equil)]
rm(data_train.subset)
data_test.11<-merge(data_test.11,data1[Semana==11,.(L4=mean(Demanda_uni_equil)),
                                       by = .(Semana,Producto_ID,Cliente_ID)],all.x=TRUE,
                    by = c("Semana","Producto_ID","Cliente_ID"))

data_test.11$L1<-data_test.11$L3
data_test.11$L1[is.na(data_test.11$L1)]<-0
data_test.11$L3<-NULL

data_test.11$L2<-data_test.11$L4
data_test.11$L2[is.na(data_test.11$L2)]<-0
data_test.11$L4<-NULL
head(data_test.11)
data_test.11 <-data_test.11[order(id),]
data_test.11.matrix <-sparse.model.matrix(~id+Semana+nCliente_ID+nProducto_ID+
                                            +L1+L2+Producto_ID+Cliente_ID,
                                          data=data_test.11,
                                          contrasts.arg = c("Producto_ID","Cliente_ID"),
                                          sparse = TRUE, sci = FALSE)

pred2<-predict(model,data_test.11.matrix)
pred2[pred2 < 0] <- 0

solution1 <- data.frame(id=data_test.matrix10[,"id"], Demanda_uni_equil=pred1)
solution2 <- data.frame(id=data_test.11.matrix[,"id"],Demanda_uni_equil=pred2)
solution <-rbind(solution1,solution2)
solution<-solution[order(solution$id),]

write.csv(solution, file = "mysolution42.csv", row.names = FALSE)