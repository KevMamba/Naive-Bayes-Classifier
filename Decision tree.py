import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from operator import itemgetter
from sklearn.metrics import confusion_matrix

def entropy(target):
    val,count = np.unique(target,return_counts=True)
    entr = np.sum([(-(count[i]/np.sum(count))*np.log2(count[i]/np.sum(count))) for i in range(len(count))])
    return(entr)

def gain(df,attr,target):
    tent = entropy(df[target])
    val,count = np.unique(df[attr],return_counts=True)
    attr_ent = np.sum([((count[i]/np.sum(count))*entropy(df.where(df[attr]==val[i]).dropna()[target])) for i in range(len(count))])
    return tent-attr_ent


def dec_tree(used_data,data,labels,target,prev_class=None):
	if (len(np.unique(used_data[target]))<=1):
		return np.unique(used_data[target])
	elif (len(used_data)==0):
		t_max_ind = np.argmax(np.unique(data[target],return_counts=True)[1])
		return np.unique(data[target])[t_max_ind]
	elif (len(labels)==0):
		#print('here')
		return prev_class
	else:
		prev_class = np.unique(used_data[target])[np.argmax(np.unique(used_data[target],return_counts=True)[1])]
		#print(np.unique(used_data[target],return_counts=True))
		attr_gain ={}
		for i in labels:
			attr_gain[i]=gain(used_data,i,target)
		best_f = max(attr_gain.items(),key=itemgetter(1))[0]
		#print(best_f)
		tree = {best_f:{}}
		labels = [i for i in labels if i!=best_f]
		#print(labels)
		for i in np.unique(used_data[best_f]):
			new_data = data.where(data[best_f]==i).dropna()
			tree[best_f][i] = dec_tree(new_data,data,labels,target,prev_class)
		return tree
    
def testing(tdata,tree,col,num):
	for i in col:
		if i in tree.keys():
			result = tree[i][tdata.get_value(num,i)]
			#print(type(result))				
			if (isinstance(result,dict)):
				return testing(tdata,result,col,num)
			else:
				return result 
def conf_mat(pred,given):
	cm = confusion_matrix(given,pred)
	acc = (cm[0][0] + cm[1][1])/(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
	prec=(cm[0][0])/(cm[0][0]+cm[0][1])
	recall=(cm[0][0])/(cm[0][0]+cm[1][0])
	return cm,acc,prec,recall
            
data = pd.read_csv('C:/Users/kevin/Desktop/PES University/Junior Year/ML/MiniProject_Section_A/House-votes-data.TXT',delimiter=',',names=['handicapped-infants','water-project-cost-sharing','adoption-of-the-budget-resolution','physician-fee-freeze','el-salvador-aid','religious-groups-in-schools','anti-satellite-test-ban','aid-to-nicaraguan-contras','mx-missile','immigration','synfuels-corporation-cutback','education-spending','superfund-right-to-sue','crime','duty-free-exports','export-administration-act-south-africa','ClassName'])
data = data.replace('?',pd.NaT)
data = data.drop(['religious-groups-in-schools','anti-satellite-test-ban','crime','duty-free-exports'],axis=1)
data_train,data_test=train_test_split(data,test_size=0.1,random_state=42)
target = 'ClassName'
labels = list(data_train.columns)
labels.remove(target)
target_class = np.unique(data_train[target])[np.argmax(np.unique(data_train[target]))]
tree = dec_tree(data_train,data_train,labels,target)
#print(tree)
x_test=data_test.drop(target,axis=1).reset_index(drop=True)
y_test=data_test[target].reset_index(drop=True)
y_test=y_test.to_frame()
cols = list(x_test.columns)
for col in cols:
	x_test[col] = x_test[col].fillna(x_test[col].mode()[0])
pred = pd.DataFrame(columns=['Model_value'])
for i in range(len(y_test)):
	#tclass = testing(x_test.iloc[[i]],tree,cols)
	#print(x_test.iloc[[i]])
	tclass = testing(x_test,tree,cols,i)
	pred.loc[i] = tclass
conmat,accuracy,precision,recall=conf_mat(pred,y_test)	
#print(pred)
#print(y_test)
print("The accuracy of the decision tree model is: ",(np.sum(pred['Model_value'].sort_index() ==y_test['ClassName'].sort_index())/len(data_test))*100)
print("The precision of the model is: ",precision*100)
print("The recall of the model is: ",recall*100)
print("The f-measure of the model is: ",((recall*precision)/(recall+precision))*200)

