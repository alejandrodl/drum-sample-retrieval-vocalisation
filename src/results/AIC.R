#setwd('/Users/alejandrodelgadoluezas/Documents/GitHub/drum-sample-recommendation-by-vocal-imitation')
library(lme4)
library(lmerTest)
for (idx in c("0","1","2","3","4")){
data <- read.table(paste(paste("results/LMER_Dataset_", idx, sep="",collapse=NULL),".csv",sep="",collapse=NULL), header=TRUE, sep=",", row.names=NULL)
AICs <- vector()
data[,2] = as.factor(data[,2])
data[,3] = as.factor(data[,3])
data[,4] = as.factor(data[,4])
data[,5] = as.factor(data[,5])
data[,6] = as.factor(data[,6])
data[,8] = as.factor(data[,8])
c <- 1
for (dist in c("Heuristic","CAEB","CAE","CAESL","CAEDL","CAESDL")){
f <- formula(paste("rating ~ (1|listener/trial) + (1|imitator) + rated_sound * ", dist))
model <- lmer(formula=f, data=data)
AICs <- append(AICs, extractAIC(model)[2])
#print(paste(sprintf("%s: %f", dist, AICs[c]-51000)))
c <- c+1
}
if (idx=="0"){
AICs_All <- AICs
}
else {
AICs_All <- AICs_All + AICs
}
}
c <- 1
for (dist in c("Heuristic","CAEB","CAE","CAESL","CAEDL","CAESDL")){
print(paste(sprintf("Mean %s: %f", dist, (AICs_All[c]/5)-51000)))
c <- c+1
}