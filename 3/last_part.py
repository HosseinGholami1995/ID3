import pandas as pd
from collections import Counter
from math import log2
from pprint import pprint


path='F:/Data/noisy_train.csv'
df_shroom=pd.read_csv(path)

path1='F:/Data/noisy_valid.csv'
df_shroom_valid=pd.read_csv(path1)

path2='F:/Data/noisy_test.csv'
df_shroom_test=pd.read_csv(path2)


def entropy(prob):
 return sum( [-p*log2(p) for p in prob] )
    
def entropy_of_list(a_list):
    cnt = Counter(x for x in a_list)
 
    num_instances = len(a_list)*1.0
    #mohasebe ehtemal ha 
    probs = [x / num_instances for x in cnt.values()]
    
    return entropy(probs)
    

def gain(feature_name):
    # dar avardan category haye featue
    df_categoty = df_shroom.groupby(feature_name)
    
    # mohasebe entropy ha dar dataframe roye target_attribute_name
    n = len(df_shroom.index) * 1.0
    df_agg_ent=df_categoty.agg({'poisonous':
            [entropy_of_list, lambda x: len(x)/n] })['poisonous']
  
    
    df_agg_ent.columns = ['entropy', 'prob']
     
    #jamh entropy ha va zarb dar ehtemaleshon
    return sum( df_agg_ent['entropy'] * df_agg_ent['prob'] )
    
#__________ta in ja be nahvi dar code aval neveshte shod ______________________

def id3(df , feature_names , depth_of_tree, last_lable=None):
    #baraye moshakhas kardan depth tree 
    #aya dade ya na
    if(depth_of_tree!=None):
        dpt=depth_of_tree-1
    else:
        dpt=None
    #bara hame lable ha :
    cnt = Counter(x for x in df['poisonous'])
    
    #sharayet tvaghof tree
    ##1:agr hame label ha az yek model bashand
    if len(cnt) == 1:
        a=list(cnt.keys())
        return a[0]
    ##2:agr feature ha tamam shode bashand ya az hameye data estefade karde bashim
    elif df.empty or (not feature_names):
        return last_lable 
     ###hala bia va derakh ra beshkan
    else:
        # label aksariyat in ja ra baraye laye badi dashte bash
        #voting 
        a=list(cnt.values())
        index_of_max = a.index(max(cnt.values())) 
        a=list(cnt.keys())
        last_lable = a[index_of_max] 
        #mohasebe gain 
        #bia gain ha ra hesab kon va kamtrin ra bardar
            #choon az entropy ghabl kam nakrdi kamtarin gain bishtarin data ra darad
        gainz = [gain(feature) for feature in feature_names]
        index_of_max_gain = gainz.index(min(gainz)) 
        
        best_feature = feature_names[index_of_max_gain]
  
     ##3:agr depth baraye tree bashad va be on reside bashad      
        if(dpt!=None):
            if(dpt<0):
                return last_lable 
        
        # ezafe kardan shakhe khali daroon dict ghabli
        tree = {best_feature:{}}
        remaining_feature_names=[i for i in feature_names if i != best_feature]
        
        
 #seda kardan recursively in algorithm ta yeki az sharayet tavaghof rokh dahad
        for feature_val, data_subset in df.groupby(best_feature):
            
            subtree = id3(data_subset,remaining_feature_names,dpt)
            
            tree[best_feature][feature_val] = subtree
        return tree


#func komaki baraye conv tree be list  
def tree_to_list(d_pandas, tree):
    
    a = list(tree.keys())
    feature = a[0]
    if d_pandas[feature] in tree[feature].keys():
        result = tree[feature][d_pandas[feature]]
        if isinstance(result, dict): 
            # tosh ye dic dg hast : derakh bayad shekaste shavad
            return tree_to_list(d_pandas, result)
        else:
            return result # this is a label
    else:
        return None

##____________________________________________________________________________
#tavabeh tamoom shodan
##____________________________________________________________________________

attribute_names = list(df_shroom.columns)
attribute_names.remove('poisonous')

tree = id3(df_shroom, attribute_names,depth_of_tree=3)

pprint(tree)

#__test of train____________________________________________________________________________

train_data  = df_shroom
#applay kardan func tree_to_list be tree va rikhtan ro  predicted_train
train_data['predicted_train'] = train_data.apply(tree_to_list,axis='columns',args=(tree,) ) 

print ( 'Accuracy of train is ' ,
       sum(train_data['poisonous']==train_data['predicted_train'])/ (len(train_data.index)*1.0)
       )
#__test of valid____________________________________________________________________________

valid_data = df_shroom_valid
valid_data['predicted_valid'] = valid_data.apply(tree_to_list,axis='columns',args=(tree,) ) 

print ( 'Accuracy of validation is ' ,
       sum(valid_data['poisonous']==valid_data['predicted_valid'])/ (len(valid_data.index)*1.0)
       )

#__test of test____________________________________________________________________________
test_data  = df_shroom_test
test_data['predicted_test'] = test_data.apply(tree_to_list,axis='columns',args=(tree,) ) 

print ( 'Accuracy of test is ' ,
       sum(test_data['poisonous']==test_data['predicted_test'])/ (len(test_data.index))
       )
#__________________________________________________________________________________














