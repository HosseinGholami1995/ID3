import pandas as pd
from math import log

def entr(x,y):# x:pos y:neg
    return -((x/(x+y))*log((x/(x+y))))-((y/(x+y))*log((y/(x+y))))


path='F:/Data/noisy_train.csv'
df=pd.read_csv(path)

entropy=list()
gain=list()
tree=list()

index = df.columns.tolist()
index.remove('poisonous')
for feather in index: 
    for types ,z in df.groupby([feather]):
        if(((0,types) in df.groupby(['poisonous',feather]).groups.keys())
          &((1,types) in df.groupby(['poisonous',feather]).groups.keys())):
            entropy.append(
                        entr(df.groupby(['poisonous',feather]).size()[0][types]
                            ,df.groupby(['poisonous',feather]).size()[1][types])
                        *
                        (df.groupby([feather]).size()[types]
                        /df.groupby([feather]).size().sum() )
                        )
        else:
            entropy.append(0)
    gain.append([sum(entropy),feather]) 
   # print(entropy)
    for types in range(0,len(entropy)):
        entropy.pop()
gain.sort()
print(gain)
root=gain[0][1]
for types in range(0,len(gain)):
    gain.pop()
#root has found 

zanjir=list()
print('____________\n')
stop=0
count =0
t=1
index.remove(root)

tree_lvl=-1

internal_counter=-1

node=root

while(stop==0):
    #mohasebe zanjir ha
    if( tree_lvl==5 ):
        stop=1
    
    if(count > 0):
   
        if(internal_counter<0):
            internal_counter=len(df.groupby([node]).size())
            tree_lvl=tree_lvl+1
        
        
        
        internal_counter=internal_counter-1
        if(node in index ):
            index.remove(node)
          
        
        if(len(zanjir[tree_lvl])==6):
            node=zanjir[internal_counter][3]
            index.remove(node)
        else:
            node=zanjir[count][3]
        
  
 
    
    count =count+1
        ##calculate zanjir
    for branch ,z1 in df.groupby([node]):        
            for feather in index:
                for types ,z2 in df.groupby([feather]):
                    if(((1,types,branch) in df.groupby(['poisonous',feather,node]).groups.keys())
                      &((0,types,branch) in df.groupby(['poisonous',feather,node]).groups.keys())):
                        entropy.append(
                                entr(df.groupby(['poisonous',feather,node]).size()[0][types][branch]
                                    ,df.groupby(['poisonous',feather,node]).size()[1][types][branch])
                                *
                                (df.groupby([feather]).size()[types]
                                /df.groupby([feather]).size().sum() )
                                )
                    else:
                        entropy.append(0)
                GAIN=sum(entropy)
                for types in range(0,len(entropy)):
                    entropy.pop()
                #label zadan 
                if(GAIN==0):
                    t=1
                else:  
                    gain.append([GAIN,feather]) 
                
            gain.sort()
            #print(branch,gain)
            try:    
                if(t==0):
                    zanjir.append((tree_lvl,node,branch,gain[0][1]))
                if(t==1):
                    zanjir.append((tree_lvl,node,branch,gain[0][1],'end'))
                t=0
        
                print(count,node,branch,gain[0][1])
                for types in range(0,len(gain)):
                    gain.pop()      
            except:
                pass


#len(list(df.groupby(['poisonous','capshape']).groups.values())[0])
#df.groupby(['poisonous','capshape']).size()