import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,median_absolute_error,r2_score
vgsalesDf = pd.read_csv('vgsales.csv')
#Rank Name Platform Year Genre Publisher NA_Sales EU_Sales JP_Sales Other_Sales Global_Sales

#滤除缺失数据
print("数据清理前：", vgsalesDf.shape)
data = vgsalesDf
data.dropna(how = "any", inplace = True)
print("滤除缺失数据：", data.shape)
#删除重复数据
data = data.drop_duplicates()
print("删除重复数据：", data.shape)

data.Year = data.Year.astype(int)

corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
#Plotting heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,linewidths=.5)
b, t = plt.ylim() # Finding the values for bottom and top
b += 0.5 
t -= 0.5 
plt.ylim(b, t) 
plt.show() 

def findBestGame(data, attr):
    maxSold = data[attr].max()
    bestGame = data[data[attr] == maxSold]
    return bestGame

bestGame = findBestGame(data, "Global_Sales")
print("全球销量最好的游戏是：")
print(bestGame[["Name", "Platform", "Year", "Genre", "Publisher", "Global_Sales"]])
bestGame = findBestGame(data, "NA_Sales")
print("北美洲销量最好的游戏是：")
print(bestGame[["Name", "Platform", "Year", "Genre", "Publisher", "NA_Sales"]])
bestGame = findBestGame(data, "EU_Sales")
print("欧洲销量最好的游戏是：")
print(bestGame[["Name", "Platform", "Year", "Genre", "Publisher", "EU_Sales"]])
bestGame = findBestGame(data, "JP_Sales")
print("日本销量最好的游戏是：")
print(bestGame[["Name", "Platform", "Year", "Genre", "Publisher", "JP_Sales"]])
bestGame = findBestGame(data, "Other_Sales")
print("其他地区销量最好的游戏是：")
print(bestGame[["Name", "Platform", "Year", "Genre", "Publisher", "Other_Sales"]])

print("\n**********************************************************************************\n")

platformList = []
platformSalesList = []
platformData = data.groupby('Platform')
maxSold = platformData.Global_Sales.max()
for groupname,grouplist in platformData:
    bestGame = findBestGame(grouplist, "Global_Sales")
    print(groupname + "全球销量最好的游戏是：")
    print(bestGame[["Name", "Year", "Genre", "Publisher", "Global_Sales"]])
    platformList.append(groupname)
    platformSalesList.append(bestGame["Global_Sales"].max())

print("\n**********************************************************************************\n")

genreList = []
genreSalesList = []
platformData = data.groupby('Genre')
maxSold = platformData.Global_Sales.max()
for groupname,grouplist in platformData:
    bestGame = findBestGame(grouplist, "Global_Sales")
    print(groupname + "全球销量最好的游戏是：")
    print(bestGame[["Name", "Year", "Platform", "Publisher", "Global_Sales"]])
    genreList.append(groupname)
    genreSalesList.append(bestGame["Global_Sales"].max())

plt.xticks(rotation = 75)
x_axis = data['Year']
sns.countplot(x= x_axis, data = data)
plt.title('Total Game Sales Each Year')
plt.show()

x = list(range(len(genreList)))
index=np.arange(len(genreList))
plt.bar(index, genreSalesList,color='steelblue',tick_label = genreList)
for a,b in zip(index, genreSalesList):
    plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=7)
plt.show()

x = list(range(len(platformList)))
index=np.arange(len(platformList))
plt.bar(index, platformSalesList,color='steelblue',tick_label = platformList)
for a,b in zip(index, platformSalesList):
    plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=7)
plt.show()

top_games_G = data.sort_values('Global_Sales',ascending = False).head(5)
matplotlib.rcParams['figure.figsize'] = (20, 10)
explode = np.zeros(len(top_games_G['Global_Sales']), dtype = float)
explode[0] = 0.1
exploded = tuple(explode)
plt.pie(top_games_G['Global_Sales'], labels = top_games_G['Name'], 
        autopct='%1.0f%%', 
        pctdistance=1.1, 
        labeldistance=1.2,explode=exploded,shadow=True)
plt.legend(bbox_to_anchor=(1,0.5), loc="center right", fontsize=15, 
           bbox_transform=plt.gcf().transFigure)
plt.show()

genreList = []
genreSalesList = []
platformData = data.groupby('Genre')
for groupname,grouplist in platformData:
    genreList.append(groupname)
    genreSalesList.append(grouplist['EU_Sales'].sum())

print(genreList)
print(genreSalesList)

explode = np.zeros(len(genreSalesList), dtype = float)
explode[0] = 0.1
exploded = tuple(explode)
plt.pie(genreSalesList, labels = genreList, 
        autopct='%1.0f%%', 
        pctdistance=1.1, 
        labeldistance=1.2,explode=exploded,shadow=True)
plt.legend(bbox_to_anchor=(1,0.5), loc="center right", fontsize=15, 
           bbox_transform=plt.gcf().transFigure)
plt.show()

data2 = data.copy()
def replace_normal(x_data):
    for i in x_data.columns:
        x_data[i]=x_data[i].factorize()[0]
    return x_data    

#x_data = data2.drop(['Name', "Global_Sales", 'Rank', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'],axis = 1)
x_data = data2.drop("Global_Sales",axis = 1)
y_data = data2["Global_Sales"]
x_data = replace_normal(x_data)
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=.3,random_state=1)
#线性回归
lr_model =LinearRegression()
lr_model.fit(x_train,y_train)
y_pred=lr_model.predict(x_test)
print("r2_score: ",r2_score(y_test,y_pred))                      
print("MSE: ",mean_squared_error(y_test,y_pred))
fig = plt.figure(figsize=(10,6))
plt.plot(range(y_test[:1000].shape[0]),y_test[:1000],color="blue", linewidth=1.5, linestyle="-")
plt.plot(range(y_test[:1000].shape[0]),y_pred[:1000],color="red", linewidth=1.5, linestyle="-.")
plt.legend(['y_test','y_pred'])
plt.show()

#随机森林
rf_model = RandomForestRegressor(n_estimators=200,min_samples_split=20,random_state=43)
rf_model.fit(x_train,y_train)
y_pred = rf_model.predict(x_test)
fig = plt.figure(figsize=(10,6))
plt.plot(range(y_test[:1000].shape[0]),y_test[:1000],color="blue", linewidth=1.5, linestyle="-")
plt.plot(range(y_test[:1000].shape[0]),y_pred[:1000],color="red", linewidth=1.5, linestyle="-.")
plt.legend(['y_test','y_pred'])
plt.show()
print("r2_score: ", r2_score(y_test,y_pred))
print("MSE: ", mean_squared_error(y_test,y_pred))