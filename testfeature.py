import pandas as pd
import numpy as np
from scipy import stats
def convert(x,encoding_dict):
    #x=str(x, encoding="utf-8")
    return encoding_dict[x]
def LoadArff(filename):
    data = arff.loadarff(filename)
    df = pd.DataFrame(data[0])
    feature_type = data[1]
    return df,feature_type
def EncodeDf(df ):
    for i in df.columns:
        if i == 'city':
            encoding_dict = {}
            idx = 0
            for j in df[i].unique():
                encoding_dict[j] = idx
                idx += 1
            df[i] = df[i].apply(convert, args=( encoding_dict,))
    return df
def OneHotEncode(df):
    clm = df.columns
    for i in clm:
        if i == 'city':
            encoding_dict = {}
            idx = 0
            for j in df[i].unique():
                encoding_dict[j] = idx
                idx += 1
            # x feature value
            lenEncode = len(df[i].unique())
            # encodestr='0'*(lenEncode-encoding_dict[x])+bin(2**encoding_dict[x])[2:]
            f = lambda x, encoding_dict: list(
                '0' * (len(encoding_dict) - encoding_dict[x] - 1) + bin(2 ** encoding_dict[x])[2:])
            newdf = pd.DataFrame(np.array(df[i].apply(f, args=(encoding_dict,)).values.tolist(), dtype='int'))
            newdf.columns = [i + x for x in encoding_dict.keys()]
            df = df.drop(i, axis=1)
            df[newdf.columns] = newdf
            # df=df.apply(float)
    return df
data = pd.read_csv('citydata34.csv',header = 1)
data.columns = ['city_year','city','year','permanent population',
                'household','employed population','secundiparity town employed population',
                'tertiary employed population','pupil population','urban area','urban built_up area',
                'road area','GDP','instituional balance','resident balance','institutional loan balance',
                'total volume of retail sales','secundiparity town production value','tertiary production value',
                'fiscal revenue','fiscal expenditure','fixed asset investment','real estate investment']
beijing = data[data['city']=='北京'][['GDP','year']]
shanghai = data[data['city']=='上海'][['GDP','year']]
shenzhen = data[data['city']=='深圳'][['GDP','year']]
data=data.drop('city_year',axis=1)
# test Shanghai first.
GDPSh=data['GDP'][data['city']=='上海'].values
samplesize=len(GDPSh)
GDPother=data['GDP'][data['city']!='上海'].values
res_shanghai=stats.ttest_ind(GDPSh,GDPother,equal_var=False).pvalue
# then test Beijing
GDPSh=data['GDP'][data['city']=='北京'].values
samplesize=len(GDPSh)
GDPother=data['GDP'][data['city']!='北京'].values
res_beijing=stats.ttest_ind(GDPSh,GDPother,equal_var=False).pvalue
# finally test Shenzhen
GDPSh=data['GDP'][data['city']=='深圳'].values
samplesize=len(GDPSh)
GDPother=data['GDP'][data['city']!='深圳'].values
res_shenzhen=stats.ttest_ind(GDPSh,GDPother,equal_var=False).pvalue
#** output:
# it shows mean of shanghai is different from the others.
# we do permutation test to show if its from same distribution
import random
idx=data.index.values
reslst=[]
for i in range(100):
    random.shuffle(idx)
    tempdata=data.iloc[idx]
    GDPSh=tempdata['GDP'].iloc[0:samplesize]
    GDPother=tempdata['GDP'].iloc[samplesize:]
    reslst.append(stats.ttest_ind(GDPSh,GDPother,equal_var=False).pvalue)
#** plot the reslst values as scatter plot or histgram
# we did a permutation test to the gdp data and find they are indeed inconsistant,
# at least when conditioning on cities, they dont come from the same distribution
# we would try to do random forest estimation.
#------------------------
#use random forest to select features.
# first encode the nominal feature of data i.e. cities into numeric values.
# then run a random forest to find the important features.
# result printed at the end.
# you may plot a histgram showing the importance of each feature. you can obtain the value from
data=EncodeDf(data)
data=data.drop('city',axis=1)
data=data.apply(lambda x: (x - np.mean(x)) / np.std(x))
from sklearn.ensemble import RandomForestRegressor
'''
regr.fit(onehotdf.drop('GDP',axis=1),onehotdf['GDP'])
regr.fit(data.drop('GDP',axis=1),data['GDP'])
regr.feature_importances_ <---- importance of each feautre.
print(regr.feature_importances_)
important_features=data.columns[np.argsort(regr.feature_importances_)[::-1][0:5]].values
print(np.sort(regr.feature_importances_)[::-1][0:5])
print(important_features)'''
'''['secundiparity town production value' 'tertiary production value'
 'total volume of retail sales' 'resident balance'
 'institutional loan balance']
 '''
# prediction on three cities
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
f_data = data.drop(columns=['GDP','year']).values   # features data
#year_ = data[data['city']=='北京']['year']
tempdata=EncodeDf(tempdata)
train_data=tempdata.iloc[idx[25:]]
test_data=tempdata.iloc[idx[0:25]]
regr = RandomForestRegressor(n_estimators=256,max_depth=9, random_state=0)

regr.fit(train_data.drop('GDP',axis=1),train_data['GDP'])
predictions = regr.predict(test_data.drop('GDP',axis=1))
newpredtor=RandomForestRegressor(n_estimators=256,max_depth=9, random_state=0)
# this block performs a cross validation of the random forest model.
# I use r2 score as the score, 10 FOLDER CROSS VALIDATION
# we can see the model is great.
from sklearn.model_selection import cross_val_score
cross_val_score(newpredtor,tempdata.drop('GDP',axis=1),tempdata['GDP'],scoring='r2')
r2score=cross_val_score(newpredtor,tempdata.drop('GDP',axis=1),tempdata['GDP'],scoring='r2',cv=10)
import matplotlib.pyplot as plt
# this block of codes do the predcition on the training set we created on the data set.
# plot it into scatter plots

plt.figure(1)
plt.scatter(test_data['year'],test_data['GDP'],label='True Value')
plt.scatter(test_data['year'],predictions, label = 'Predicted Value')
plt.legend()
plt.close()
# then try to predict estimate the model only using the selected features.
RealNewPreditor=RandomForestRegressor(n_estimators=256,max_depth=9, random_state=0)
r2scoreSelected=cross_val_score(RealNewPreditor,tempdata[['secundiparity town production value', 'tertiary production value',
 'total volume of retail sales' ,'resident balance',
 'institutional loan balance']],tempdata['GDP'],scoring='r2',cv=10)
print(r2scoreSelected)
regr.predict()

pass



