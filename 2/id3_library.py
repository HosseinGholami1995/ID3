import pandas as pd

from id3 import Id3Estimator
from id3 import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#from id3 import export_text

path='F:/Data/noisy_train.csv'
tr=pd.read_csv(path)


#path1='F:/Data/noisy_valid'
path1='F:/Data/noisy_test.csv'


va=pd.read_csv(path1)


feature_names = tr.columns.tolist()
feature_names.remove('poisonous')
#__________________________________________________

Xtr=tr.values.tolist()
for i in range(0,len(tr.values)):
    del Xtr[i][0]
ytr=list(tr['poisonous'])
#_____________
Xva=va.values.tolist()
for i in range(0,len(va.values)):
    del Xva[i][0]
yva=list(va['poisonous'])
#__________________________________________________

Xtr_train, Xtr_test, ytr_train, ytr_test = train_test_split(Xtr, ytr
                                                ,test_size=0.0,shuffle=False)

Xva_train, Xva_test, yva_train, yva_test = train_test_split(Xva, yva
                                                ,test_size=0.0,shuffle=False)
#______________________________<< start learning >>_________________________________________________

estimator = Id3Estimator(max_depth=2, min_samples_split=1, prune=False,
            gain_ratio=False, min_entropy_decrease=0, is_repeating=False)

estimator.fit(Xtr_train, ytr_train, check_input=False)
export_graphviz(estimator.tree_, 'tree_d2.dot', feature_names,extensive=True)

y_predtr=estimator.predict(Xtr_train)


y_predva=estimator.predict(Xva_train)


print('Accuracy Score on train data: ', 
      accuracy_score(y_true = ytr_train  , y_pred=y_predtr )
      )
print('Accuracy Score on test data: ',
      accuracy_score(y_true=yva_train,y_pred=y_predva)
      )

