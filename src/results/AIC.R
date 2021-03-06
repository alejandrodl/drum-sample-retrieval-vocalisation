#setwd('/Users/alejandrodelgadoluezas/Documents/GitHub/drum-sample-retrieval-vocalisation')
library(lme4)
library(lmerTest)
AICs_All <- matrix(0, 5, 8)
c1 <- 1
for (idx in c("0","1","2","3","4")){
data <- read.table(paste(paste("results/LMER_Dataset_", idx, sep="",collapse=NULL),".csv",sep="",collapse=NULL), header=TRUE, sep=",", row.names=NULL)
AICs <- vector()
data[,2] = as.factor(data[,2])
data[,3] = as.factor(data[,3])
data[,4] = as.factor(data[,4])
data[,5] = as.factor(data[,5])
data[,6] = as.factor(data[,6])
data[,8] = as.factor(data[,8])
for (dist in c("Random","Heuristic","CAEBOriginal","CAEB","CAE","CAESL","CAEDL","CAESDL")){
f <- formula(paste("rating ~ (1|listener/trial) + (1|imitator) + rated_sound * ", dist))
model <- lmer(formula=f, data=data)
AICs <- append(AICs, extractAIC(model)[2])
}
AICs_All[c1,] <- AICs
c1 <- c1+1
}
c2 <- 1
for (dist in c("Random","Heuristic","CAEBOriginal","CAEB","CAE","CAESL","CAEDL","CAESDL")){
print(paste(sprintf("AIC %s: %f +- %f", dist, mean(AICs_All[,c2])-51015, sd(AICs_All[,c2])*(1.96/(sqrt(5))))))
c2 <- c2+1
}