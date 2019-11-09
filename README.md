# Predicting-stock-return-volatility-with-machine-learning
Code used for my Master Thesis

This code was used for my Master Thesis. It predicts the stock return volatility with the help of the following deep learning networks: 
MLP - Multilayer Perceptron
JORDAN - Jordan Network
ELMAN - Elman Network
LSTM - Long short term memory

Before running the Neural Network, the data needs to be standardized in terms of date and merged afterwards. In this thesis unemployment rate and GNP has been used as explainatory variables. 
```{r}
stock<-getSymbols("^GSPC",auto.assign = FALSE, from =min("2018-01-01"),to=Sys.Date())
daily_return<-Return.calculate(stock$GSPC.Close,method=c("log"))
stock<-data.frame(index(stock),daily_return)
colnames(stock)<-c("date","daily_return")
stock<-merge(data,stock)
stock<-stock%>%
  mutate(
    month_year=paste0(month(stock$date),year(stock$date)))
#### Unemployment Rate
uep<-read.csv("UNRATE.csv")
uep$DATE<-as.Date(uep$DATE,format="%Y-%m-%d")
uep$UNRATE<-as.numeric(as.character(uep$UNRATE))
uep<-uep[lubridate::year(uep$DATE)<=2017,]
colnames(uep)<-tolower(colnames(uep))
uep<-uep%>%
  mutate(
    month_year=paste0(month(uep$date),year(uep$date)))
#### GNP
gnp<-read.csv("GNP.csv")
gnp$DATE<-as.Date(gnp$DATE,format="%Y-%m-%d")
gnp$GNP<-as.numeric(as.character(gnp$GNP))
gnp<-gnp[lubridate::year(gnp$DATE)<=2017,]
colnames(gnp)<-tolower(colnames(gnp))
gnp<-data.frame(gnp,month_year=paste0(month(gnp$date),year(gnp$date)))
## Weekly RV
rt_w<-rep(colMeans(matrix(stock$daily_return,5)),each=5)
rt_m<-stock %>%
  group_by(month_year) %>%
  summarise(mean = mean(daily_return))
#### merging
stock<-stock%>%
  left_join(uep,by="month_year")%>%
  left_join(gnp,by="month_year")%>%
  left_join(rt_m,by="month_year")%>%
  fill(gnp)%>%
  fill(date)%>%
  cbind(rt_w[1:nrow(stock)])
```
The special case of this Neural Network is, that is uses the predicted values of the traditional time series modesl. Thus the forecast of the ARCH, GARCH and GARCH-MIDAS are included into the Neural Network.
```{r}
colnames(stock)<-c("date","rvol","daily_returns","month_year_paste","date_month","uemp","date_quarter","gnp","rt_m","rt_w")
stock<-stock[2:nrow(stock),]#remove na from daily returns

egarch_fc<-read.csv("egarch_window_fc_complete.csv")
garch_fc<-read.csv("garch_window_fc_complete.csv")
egarch_fc<-egarch_fc$x**2
garch_fc<-garch_fc$x**2
gmidas_fc<-read.csv("gmidas.csv")[1:(nrow(stock)-254),]
gmidas_xfc<-read.csv("gmidasX.csv")[1:(nrow(stock)-254),]

stock<-cbind(stock[255:nrow(stock),],garch_fc,egarch_fc,gmidas_fc,gmidas_xfc)

data_range<-function(x) {(x-min(x))/(max(x)-min(x))}

stock$rt_w[is.na(stock$rt_w)==T]<-0
```
The Neural Network Model not only does include the traditional output, but also different mean model for the volatility are consierd. One model includes the ARMA residuals and the other no mean model.
```{r}
xt7<-stock$daily_returns[1:(length(stock$daily_returns)-7)]**2
xt6<-stock$daily_returns[2:(length(stock$daily_returns)-6)]**2
xt5<-stock$daily_returns[3:(length(stock$daily_returns)-5)]**2
xt4<-stock$daily_returns[4:(length(stock$daily_returns)-4)]**2
xt3<-stock$daily_returns[5:(length(stock$daily_returns)-3)]**2
xt2<-stock$daily_returns[6:(length(stock$daily_returns)-2)]**2
xt1<-stock$daily_returns[7:(length(stock$daily_returns)-1)]**2
stock<-cbind(stock[8:nrow(stock),],x=stock$rvol[8:nrow(stock)],xt1,xt2,xt3,xt4,xt5,xt6,xt7)

# resid uncond mean
#xt7<-(mean(stock$daily_returns)-stock$daily_returns[1:(length(stock$daily_returns)-7)])**2
#xt6<-(mean(stock$daily_returns)-stock$daily_returns[2:(length(stock$daily_returns)-6)])**2
#xt5<-(mean(stock$daily_returns)-stock$daily_returns[3:(length(stock$daily_returns)-5)])**2
#xt4<-(mean(stock$daily_returns)-stock$daily_returns[4:(length(stock$daily_returns)-4)])**2
#xt3<-(mean(stock$daily_returns)-stock$daily_returns[5:(length(stock$daily_returns)-3)])**2
#xt2<-(mean(stock$daily_returns)-stock$daily_returns[6:(length(stock$daily_returns)-2)])**2
#xt1<-(mean(stock$daily_returns)-stock$daily_returns[7:(length(stock$daily_returns)-1)])**2
#stock<-cbind(stock[8:nrow(stock),],x=stock$rvol[8:nrow(stock)],xt1,xt2,xt3,xt4,xt5,xt6,xt7)

# resid arma
arimaresid<-auto.arima(stock$daily_returns)
arimaresid<-arimaresid$residuals
xt7<-arimaresid[1:(length(stock$daily_returns)-7)]**2
xt6<-arimaresid[2:(length(stock$daily_returns)-6)]**2
xt5<-arimaresid[3:(length(stock$daily_returns)-5)]**2
xt4<-arimaresid[4:(length(stock$daily_returns)-4)]**2
xt3<-arimaresid[5:(length(stock$daily_returns)-3)]**2
xt2<-arimaresid[6:(length(stock$daily_returns)-2)]**2
xt1<-arimaresid[7:(length(stock$daily_returns)-1)]**2
stock<-cbind(stock[8:nrow(stock),],x=stock$rvol[8:nrow(stock)],xt1,xt2,xt3,xt4,xt5,xt6,xt7)
```
This is where the fun begins! To model the Neural Network for LSTM, the Keras package is used. Before the data is fed to the algorithm, it needs to be preprocessed. In this case i am using the Min-Max method. This code runs several Neural Network with different inputs. The output of those models are saved in a csv data. However the total process is very time consuming and can take up to 10 hours!.
```{r}
#########################################
lag_dr_df<-stock%>%
  select(-"date",-"month_year_paste",-"date_month",-"date_quarter")

lag_dr_range<-as.matrix(sapply(lag_dr_df,data_range))



unscale_data<-function(x,max_x,min_x){x*(max_x-min_x)+min_x}
softplus <- function(x) log(1+exp(x))
sigmoid = function(x) {1 / (1 + exp(-x))}
#https://stackoverflow.com/questions/34532878/package-neuralnet-in-r-rectified-linear-unit-relu-activation-function
###################################################################### standard
#####################################################################
#####################################################################
split<-initial_time_split(lag_dr_range,prop=0.9)
nn_train<-training(split)
nn_test<-testing(split)

data_input_1<-c("xt1","xt2","xt3")
data_input_2<-c("xt1","xt2","xt3","uemp")
data_input_3<-c("xt1","rt_m","rt_w")
data_input_4<-c("xt1","xt2","xt3","garch_fc")
data_input_5<-c("xt1","xt2","xt3","egarch_fc")
data_input_6<-c("xt1","xt2","xt3","garch_fc","uemp")
data_input_7<-c("xt1","xt2","xt3","egarch_fc","uemp")
data_input_8<-c("xt1","xt2","xt3","egarch_fc","garch_fc")
data_input_9<-c("xt1","xt2","xt3","egarch_fc","garch_fc","uemp")
data_input_10<-c("xt1","xt2","xt3","gmidas_xfc")
data_input_11<-c("xt1","xt2","xt3","gmidas_xfc","garch_fc")
data_input_12<-c("xt1","xt2","xt3","gmidas_xfc","egarch_fc")
data_input_13<-c("xt1","xt2","xt3","gmidas_xfc","egarch_fc","garch_fc")
data_input_14<-c("garch_fc")
data_input_15<-c("egarch_fc")
data_input_16<-c("gmidas_xfc")
listfc<-list(data_input_1,data_input_2,data_input_3,data_input_4,
             data_input_5,data_input_6,data_input_7,data_input_8,
             data_input_9,data_input_10,data_input_11,data_input_12
             ,data_input_13,data_input_14,data_input_15,data_input_16)  

result_mlp<-c()
errors_mlp<-list()
ptm <- proc.time()
for(i in 1:16){
  set.seed(8)
  modelmlp<-neuralnet(as.formula(paste("x",paste(listfc[[i]], collapse="+"),sep=" ~ ")),data=nn_train, hidden = c(5),act.fct =softplus,learningrate=0.1,
                        err.fct = "sse", linear.output=TRUE,threshold = 0.01,stepmax = 1e+07,rep=1)
  pred<-compute(modelmlp, nn_test)
  spl<-compute(modelmlp, nn_train)
  
  mlp_actual<-unscale_data(pred$net.result,max(stock[,c("xt1")]),min(stock[,c("xt1")]))
  result_mlp[i]<-format(rmse(mlp_actual,stock[(nrow(stock)-nrow(nn_test)+1):nrow(stock),c("rvol")]),scientific = FALSE)
  errors_mlp[[i]]<-stock[(nrow(stock)-nrow(nn_test)+1):nrow(stock),c("rvol")]-mlp_actual
  }
proc.time() - ptm

write.csv(result_mlp,"result_mlp_cm.csv",row.names = F)
write.csv(errors_mlp,"errors_mlp_cm.csv",row.names = F)
#10687.2

result_elman<-c()
errors_elman<-list()
ptm <- proc.time()
for(i in 1:16){
set.seed(8)
modelelman<-elman(nn_train[,listfc[[i]]],nn_train[,c("x")],size=c(5),learnFuncParams=c(0.1),maxit=1000,
                   inputsTest = nn_test[,listfc[[i]]], updateFunc = "JE_Order",learnFunc = "JE_Rprop",
                   targetsTest = nn_test[,c("x")])


elman_actual<-unscale_data(modelelman$fittedTestValues,max(stock[,c("xt1")]),min(stock[,c("xt1")]))
result_elman[i]<-format(rmse(elman_actual,stock[(nrow(stock)-nrow(nn_test)+1):nrow(stock),c("rvol")]),scientific = FALSE)
errors_elman[[i]]<-stock[(nrow(stock)-nrow(nn_test)+1):nrow(stock),c("rvol")]-elman_actual
}
proc.time() - ptm


result_jordan<-c()
errors_jordan<-list()
ptm <- proc.time()
for(i in 1:16){
  set.seed(8)
  modelelman<-jordan(nn_train[,listfc[[i]]],nn_train[,c("x")],size=c(5),learnFuncParams=c(0.1),maxit=1000,
                    inputsTest = nn_test[,listfc[[i]]], updateFunc = "JE_Order",learnFunc = "JE_Rprop",
                    targetsTest = nn_test[,c("x")])
  
  
  jordan_actual<-unscale_data(modelelman$fittedTestValues,max(stock[,c("xt1")]),min(stock[,c("xt1")]))
  result_jordan[i]<-format(rmse(jordan_actual,stock[(nrow(stock)-nrow(nn_test)+1):nrow(stock),c("rvol")]),scientific = FALSE)
  errors_jordan[[i]]<-stock[(nrow(stock)-nrow(nn_test)+1):nrow(stock),c("rvol")]-jordan_actual
  }
proc.time() - ptm


write.csv(result_elman,"result_elman_um.csv",row.names = F)
write.csv(result_jordan,"result_jordan_um.csv",row.names = F)
write.csv(errors_elman,"errors_elman_um.csv",row.names = F)
write.csv(errors_jordan,"errors_jordan_um.csv",row.names = F)

############################################################### Plotting ############################################### 
################

use_session_with_seed(8)
errors_lstm<-list()
result_lstm<-c()
db_lstm<-c()
ptm <- proc.time()
for (i in 1:16){
xkeras_train<-nn_train[,listfc[[i]]]
ykeras_train<-nn_train[,c("x")]

xkeras_test<-nn_test[,listfc[[i]]]
ykeras_test<-nn_test[,c("x")]

time=ifelse(is.null(nrow(xkeras_train))==T,length(xkeras_train),nrow(xkeras_train))
timey=ifelse(is.null(nrow(xkeras_test))==T,length(xkeras_test),nrow(xkeras_test))
dim(xkeras_train)<-c(time,ifelse(is.null(ncol(xkeras_train))==T,1,ncol(xkeras_train)),1)
dim(xkeras_test)<-c(timey,ifelse(is.null(ncol(xkeras_test))==T,1,ncol(xkeras_train)),1)
model<-keras_model_sequential()
model%>%
  layer_lstm(5,input_shape = c(ncol(xkeras_train),1),activation="relu")%>%
   #layer_dense(units=25,activation="relu")%>%
   #layer_dense(units=5)%>%
  layer_dense(units=1,activation="linear")
 
model%>%compile(
   loss="mae",
   optimizer="RMSprop",
   metrics=c("mae") 
)

model%>%fit(xkeras_train,ykeras_train,epochs=50,batch_size=256,shuffle=F)
  
#model%>%evaluate(xkeras_train,ykeras_train,verbose=0)
#model%>%evaluate(xkeras_test,ykeras_test,verbose=0)
y_pred=model%>%predict(xkeras_test)
lstm_actual<-unscale_data(y_pred,max(stock[,c("xt1")]),min(stock[,c("xt1")]))
result_lstm[i]<-format(rmse(lstm_actual,stock[(nrow(stock)-nrow(nn_test)+1):nrow(stock),c("rvol")]),scientific = FALSE)
errors_lstm[[i]]<-stock[(nrow(stock)-nrow(nn_test)+1):nrow(stock),c("rvol")]-lstm_actual
}
proc.time() - ptm

write.csv(result_lstm,"result_lstm_cm.csv",row.names = F)
write.csv(errors_lstm,"errors_lstm_cm.csv",row.names = F)

```
