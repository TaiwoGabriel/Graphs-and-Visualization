import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


#NBE
NBE_data = "C:/Users/Gabriel/Desktop/latex_to_python2/NBE.tex"
NBE = pd.read_csv(NBE_data,sep='&',header=None)

print(NBE.to_string(), "\n")

NBE11 = NBE.iloc[:, 0:1]
#print(df11)


NBE1 = NBE.iloc[0:2, 2:12]
NBE_test2 = NBE1.iloc[0].tolist()
NBE_test = sorted(NBE_test2)
NBE_train = NBE1.iloc[1].tolist()


NBE_df1 = pd.DataFrame(NBE_test)
NBE_df2 = pd.DataFrame(NBE_train)
NBE_df3 = pd.concat([NBE_df1, NBE_df2], axis=1)
NBE_df = NBE_df3.transpose()
NBE2 = NBE_df.reset_index(drop=True)
print(NBE2)

#NBE2 = NBE1.apply(lambda x: np.sort(x), axis=1, raw=True)
#print("Sorted Dataframe", "\n")


NBE2.loc["2"] = 1-NBE2.iloc[0]
NBE2.loc["3"] = 1-NBE2.iloc[1]
NBE2.loc["4"] = NBE2.iloc[2]/NBE2.iloc[3]
NBE3 = round(NBE2,3)
#print(NBE3, "\n")

col = ["Testing accuracy", "Training accuracy", "Testing error", "Training error", "GF"]
NBE3.insert(loc=0, column='1', value=col)
#print(df3.to_string())

col2 = NBE11.iloc[0]
NBE3.insert(loc=0, column='0', value=col2)
NBE4 = NBE3.replace(np.nan, '')
NBE4.set_index('0', inplace=True)
print(NBE4.to_string())
print("\n")



#kNNE
kNNE_data = "C:/Users/Gabriel/Desktop/latex_to_python2kNNE.tex"
kNNE = pd.read_csv(kNNE_data,sep='&',header=None)

print(kNNE.to_string(), "\n")

kNNE11 = kNNE.iloc[:, 0:1]
#print(df11)


kNNE1 = kNNE.iloc[0:2, 2:12]
kNNE_test2 = kNNE1.iloc[0].tolist()
kNNE_test = sorted(kNNE_test2)
kNNE_train = kNNE1.iloc[1].tolist()


kNNE_df1 = pd.DataFrame(kNNE_test)
kNNE_df2 = pd.DataFrame(kNNE_train)
kNNE_df3 = pd.concat([kNNE_df1, kNNE_df2], axis=1)
kNNE_df = kNNE_df3.transpose()
kNNE2 = kNNE_df.reset_index(drop=True)
print(kNNE2)

#kNNE2 = kNNE1.apply(lambda x: np.sort(x), axis=1, raw=True)
#print("Sorted Dataframe", "\n")


kNNE2.loc["2"] = 1-kNNE2.iloc[0]
kNNE2.loc["3"] = 1-kNNE2.iloc[1]
kNNE2.loc["4"] = kNNE2.iloc[2]/kNNE2.iloc[3]
kNNE3 = round(kNNE2,3)
#print(kNNE3, "\n")

col = ["Testing accuracy", "Training accuracy", "Testing error", "Training error", "GF"]
kNNE3.insert(loc=0, column='1', value=col)
#print(df3.to_string())

col2 = kNNE11.iloc[0]
kNNE3.insert(loc=0, column='0', value=col2)
kNNE4 = kNNE3.replace(np.nan, '')
kNNE4.set_index('0', inplace=True)
print(kNNE4.to_string())
print("\n")




#DTE
DTE_data = "C:/Users/Gabriel/Desktop/latex_to_python2/DTE.tex"
DTE = pd.read_csv(DTE_data,sep='&',header=None)

print(DTE.to_string(), "\n")

DTE11 = DTE.iloc[:, 0:1]
#print(df11)


DTE1 = DTE.iloc[0:2, 2:12]
DTE_test2 = DTE1.iloc[0].tolist()
DTE_test = sorted(DTE_test2)
DTE_train = DTE1.iloc[1].tolist()


DTE_df1 = pd.DataFrame(DTE_test)
DTE_df2 = pd.DataFrame(DTE_train)
DTE_df3 = pd.concat([DTE_df1, DTE_df2], axis=1)
DTE_df = DTE_df3.transpose()
DTE2 = DTE_df.reset_index(drop=True)
print(DTE2)

#DTE2 = DTE1.apply(lambda x: np.sort(x), axis=1, raw=True)
#print("Sorted Dataframe", "\n")


DTE2.loc["2"] = 1-DTE2.iloc[0]
DTE2.loc["3"] = 1-DTE2.iloc[1]
DTE2.loc["4"] = DTE2.iloc[2]/DTE2.iloc[3]
DTE3 = round(DTE2,3)
#print(DTE3, "\n")

col = ["Testing accuracy", "Training accuracy", "Testing error", "Training error", "GF"]
DTE3.insert(loc=0, column='1', value=col)
#print(df3.to_string())

col2 = DTE11.iloc[0]
DTE3.insert(loc=0, column='0', value=col2)
DTE4 = DTE3.replace(np.nan, '')
DTE4.set_index('0', inplace=True)
print(DTE4.to_string())
print("\n")




#RF
RF_data = "C:/Users/Gabriel/Desktop/latex_to_python2RF.tex"
RF = pd.read_csv(RF_data,sep='&',header=None)

print(RF.to_string(), "\n")

RF11 = RF.iloc[:, 0:1]
#print(df11)


RF1 = RF.iloc[0:2, 2:12]
RF_test2 = RF1.iloc[0].tolist()
RF_test = sorted(RF_test2)
RF_train = RF1.iloc[1].tolist()


RF_df1 = pd.DataFrame(RF_test)
RF_df2 = pd.DataFrame(RF_train)
RF_df3 = pd.concat([RF_df1, RF_df2], axis=1)
RF_df = RF_df3.transpose()
RF2 = RF_df.reset_index(drop=True)
print(RF2)

#RF2 = RF1.apply(lambda x: np.sort(x), axis=1, raw=True)
#print("Sorted Dataframe", "\n")


RF2.loc["2"] = 1-RF2.iloc[0]
RF2.loc["3"] = 1-RF2.iloc[1]
RF2.loc["4"] = RF2.iloc[2]/RF2.iloc[3]
RF3 = round(RF2,3)
#print(RF3, "\n")

col = ["Testing accuracy", "Training accuracy", "Testing error", "Training error", "GF"]
RF3.insert(loc=0, column='1', value=col)
#print(df3.to_string())

col2 = RF11.iloc[0]
RF3.insert(loc=0, column='0', value=col2)
RF4 = RF3.replace(np.nan, '')
RF4.set_index('0', inplace=True)
print(RF4.to_string())
print("\n")



#SVME
SVME_data = "C:/Users/Gabriel/Desktop/latex_to_python2/SVME.tex"
SVME = pd.read_csv(SVME_data,sep='&',header=None)

print(SVME.to_string(), "\n")

SVME11 = SVME.iloc[:, 0:1]
#print(df11)


SVME1 = SVME.iloc[0:2, 2:12]
SVME_test2 = SVME1.iloc[0].tolist()
SVME_test = sorted(SVME_test2)
SVME_train = SVME1.iloc[1].tolist()


SVME_df1 = pd.DataFrame(SVME_test)
SVME_df2 = pd.DataFrame(SVME_train)
SVME_df3 = pd.concat([SVME_df1, SVME_df2], axis=1)
SVME_df = SVME_df3.transpose()
SVME2 = SVME_df.reset_index(drop=True)
print(SVME2)

#SVME2 = SVME1.apply(lambda x: np.sort(x), axis=1, raw=True)
#print("Sorted Dataframe", "\n")


SVME2.loc["2"] = 1-SVME2.iloc[0]
SVME2.loc["3"] = 1-SVME2.iloc[1]
SVME2.loc["4"] = SVME2.iloc[2]/SVME2.iloc[3]
SVME3 = round(SVME2,3)
#print(SVME3, "\n")

col = ["Testing accuracy", "Training accuracy", "Testing error", "Training error", "GF"]
SVME3.insert(loc=0, column='1', value=col)
#print(df3.to_string())

col2 = SVME11.iloc[0]
SVME3.insert(loc=0, column='0', value=col2)
SVME4 = SVME3.replace(np.nan, '')
SVME4.set_index('0', inplace=True)
print(SVME4.to_string())
print("\n")




#NNE
NNE_data = "C:/Users/Gabriel/Desktop/latex_to_python2NNE.tex"
NNE = pd.read_csv(NNE_data,sep='&',header=None)

print(NNE.to_string(), "\n")

NNE11 = NNE.iloc[:, 0:1]
#print(df11)


NNE1 = NNE.iloc[0:2, 2:12]
NNE_test2 = NNE1.iloc[0].tolist()
NNE_test = sorted(NNE_test2)
NNE_train = NNE1.iloc[1].tolist()


NNE_df1 = pd.DataFrame(NNE_test)
NNE_df2 = pd.DataFrame(NNE_train)
NNE_df3 = pd.concat([NNE_df1, NNE_df2], axis=1)
NNE_df = NNE_df3.transpose()
NNE2 = NNE_df.reset_index(drop=True)
print(NNE2)

#NNE2 = NNE1.apply(lambda x: np.sort(x), axis=1, raw=True)
#print("Sorted Dataframe", "\n")


NNE2.loc["2"] = 1-NNE2.iloc[0]
NNE2.loc["3"] = 1-NNE2.iloc[1]
NNE2.loc["4"] = NNE2.iloc[2]/NNE2.iloc[3]
NNE3 = round(NNE2,3)
#print(NNE3, "\n")

col = ["Testing accuracy", "Training accuracy", "Testing error", "Training error", "GF"]
NNE3.insert(loc=0, column='1', value=col)
#print(df3.to_string())

col2 = NNE11.iloc[0]
NNE3.insert(loc=0, column='0', value=col2)
NNE4 = NNE3.replace(np.nan, '')
NNE4.set_index('0', inplace=True)
print(NNE4.to_string())
print("\n")


# Plotting graph
plt.figure(figsize=(20, 15))
#plt.tight_layout(pad=8.0)


class_distr = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]

#NBE
x1 =  NBE_test
y1 = NBE_train


#print(x1, '\n')
plt.subplot(4, 4, 1)
plt.plot(class_distr,x1,  label='Testing Accuracy')
plt.plot(class_distr,y1,  label='Training Accuracy')
plt.xlabel('Bag Size', fontsize=20)
plt.ylabel('Accuracy ', fontsize=20)
plt.tick_params(labelsize=12, axis='both')
plt.ylim(0,1)
plt.grid(True)
plt.legend()
plt.title('NBE', fontsize=20)


#kNNE:
x2 =  kNNE_test
y2 = kNNE_train

#print(x2, '\n')
plt.subplot(4, 4, 2)
plt.plot(class_distr,x2,  label='Testing Accuracy')
plt.plot(class_distr,y2,  label='Training Accuracy')
plt.xlabel('Bag Size', fontsize=20)
plt.tick_params(labelsize=12, axis='both')
plt.ylim(0,1)
#plt.yticks([])
plt.grid(True)
plt.legend()
plt.title('kNNE', fontsize=20)

#DTE:
x3 =  DTE_test
y3 = DTE_train

#print(x3, '\n')
plt.subplot(4, 4, 3)
plt.plot(class_distr,x3,  label='Testing Accuracy')
plt.plot(class_distr,y3,  label='Training Accuracy')
plt.xlabel('Bag Size', fontsize=20)
plt.tick_params(labelsize=12, axis='both')
plt.ylim(0,1)
plt.grid(True)
plt.legend()
plt.title('DTE', fontsize=20)


#RF:
x4 =  RF_test
y4 = RF_train

#print(x4, '\n')
plt.subplot(4, 4, 4)
plt.plot(class_distr,x4,  label='Testing Accuracy')
plt.plot(class_distr,y4,  label='Training Accuracy')
plt.xlabel('Bag Size', fontsize=20)
plt.tick_params(labelsize=12, axis='both')
plt.ylim(0,1)
plt.grid(True)
plt.legend()
plt.title('RF', fontsize=20)


#SVME:
x5 =  SVME_test
y5 = SVME_train

#print(x5, '\n')
plt.subplot(4, 4, 5)
plt.plot(class_distr,x5,  label='Testing Accuracy')
plt.plot(class_distr,y5,  label='Training Accuracy')
plt.xlabel('Bag Size', fontsize=20)
plt.ylabel('Accuracy ', fontsize=20)
plt.tick_params(labelsize=12, axis='both')
plt.ylim(0,1)
plt.grid(True)
plt.legend()
plt.title('SVME', fontsize=20)


#NNE:
x6 =  NNE_test
y6 = NNE_train

#print(x6, '\n')
plt.subplot(4, 4, 6)
plt.plot(class_distr,x6,  label='Testing Accuracy')
plt.plot(class_distr,y6, label='Training Accuracy')
plt.xlabel('Bag Size', fontsize=20)
plt.tick_params(labelsize=12, axis='both')
plt.ylim(0,1)
plt.grid(True)
plt.legend()
plt.title('NNE', fontsize=20)


plt.tight_layout(4.0)
plt.show()


