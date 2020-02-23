"""
Created on Tue Apr 17 17:40:04 2018
@author: zhongzhi
"""
#! /usr/bin/env python
# -*- coding:utf-8 -*-
#!/usr/bin/env python
#coding=utf-8

import pandas as pd
import networkx as nx
from copy import deepcopy

#====
EPISODE = 3
NET_EMBED_DIM = 64
#====

cases = pd.read_csv('C:/DATA/selfharm_patients_records.csv',na_filter=False)
cases = cases[cases['AGE']!='nnnulll']
cases['AGE'] = cases['AGE'].astype(float)
cases = cases[cases['AGE']>10]
print (cases.shape[0])

# evidence of historical episodes equal or below 3
from collections import Counter
allpatid = Counter(list(cases['PATID']))
allpatid = list(allpatid.values())
allpid_above_four = [i for i in allpatid if i>4]
print (len(allpid_above_four)/len(allpatid))

def whether_a_case(df): ##this df has sf record deleted
    if df.iloc[0]['timelength'] <= 365:
        if diags.shape[0]<=6 : return df, True
        if diags.shape[0]>6  : return df[:6], True
    else: return False, False    
    
individuals = list(set(list(cases['PATID'])))
cases_diaglist, sexlist, agelist, pidlist = [], [], [], []
for i in individuals[:]:
    records = cases[cases['PATID'] == i]
    if records.shape[0] >= 2: ##records including sf record. Patients without historical diagnoses will be removed
        records = records.sort_values(axis = 0, ascending = False, by = 'since2007')
        records.index = range(records.shape[0])
        timelength = []
        for p in records.index[1:2]:
            timelength = records.loc[0]['since2007'] - records.loc[p]['since2007'] 
        records['timelength'] = timelength    
        records = records.drop(0) ##drop first line, i.e., sf record
        
        if records.iloc[0]['diag_cd_01'] != 'null' :
            diags = pd.DataFrame(columns=('diags', 'timelength'))
            for j in records.index:
                for m in range(1, 7):
                    if records.loc[j]['diag_cd_0'+str(m)] != 'null':
                        diags.loc[diags.shape[0]] = [records.loc[j]['diag_cd_0'+str(m)], records.loc[j]['timelength']]
            a,b = whether_a_case(diags)
            if (b and (list(a['diags']) not in cases_diaglist)): 
                cases_diaglist.append(list(a['diags']))
                agelist.append(records.iloc[0]['AGE'])
                sexlist.append(records.iloc[0]['SEX'])
                pidlist.append(records.iloc[0]['PATID'])           
len(list(set(pidlist)))  #2323        

f=open('age_cases','w')
for i in agelist:
    f.write(str(i)+'\n')
f.close()
f=open('sex_cases','w')
for i in sexlist:
    if i=="b'F'":
        f.write('0'+'\n')
    else:
        f.write('1'+'\n')
f.close()


#========
all_case = []
individuals =list(set(list( pidlist)))
for i in individuals[:]:
    ep = dict()
    for k in range(EPISODE): ep[k] = []

    records = cases[cases['PATID'] == i]
    records = records.sort_values(axis = 0, ascending = False, by = 'since2007')
    records.index = range(records.shape[0])
    records = records.drop(0) ##drop first line, i.e., sf record
    timeline = set(list(records['since2007']))
    
    count1 = 0 #count the # of episode
    for t in timeline:
        diags_in_an_episode = records[records['since2007'] == t]
        diags_in_an_episode = diags_in_an_episode[['diag_cd_01','diag_cd_02','diag_cd_03','diag_cd_04','diag_cd_05','diag_cd_06']].values
        diags_in_an_episode = list(diags_in_an_episode[0,:])
        diags_in_an_episode = list(filter(lambda x : x != 'null', diags_in_an_episode))  
        
        count2=0 #count the # of diagnoses in an episode
        for j in diags_in_an_episode:
            ep[count1].append(j)
            if count2>=2: break
            count2=count2+1    
        if count1>=2: break
        count1 = count1+1
    all_case.append(ep)
 
#==FIRST THREE DIGIT
temp2= []
for i in all_case:
    for j in i.keys():
        for d in i[j]:
            l = d.strip().split('.')
            temp2.append(l[0])
        i[j] = temp2
        temp2=[]

#==nodes in network are not coded in icd, ]
#==so build a dict to traslate    
f=open('C:/DATA/emb_selfharm.csv')
line=f.readline()
node2vec = dict()
nodes_in_net = []
while True:
    line=f.readline()
    if not line: break
    l=line.strip().split(' ')
    node2vec[l[0]] = l[1:]
    nodes_in_net.append(l[0])
f.close()   

#==for nodes not in the network, assign a zero vector 
for i in range(1200):
    if str(i) not in node2vec.keys():
        node2vec[str(i)] = [0]*NET_EMBED_DIM

#==
icd_netnode = dict()
f=open('C:/DATA/icd_category.csv','r')
line = f.readline()
while True:
    line = f.readline()
    if not line:break
    l = line.strip().split(',')
    icd_netnode[l[0]] = l[1]

#==vectors[patient_indx][episode] = a 64*3 matrix
#all_case[patient_indx][episode][main/second/third_diagnosis]
m,n = len(list(set(pidlist))),3
vectors = [[0 for i in range(n)] for j in range(m)]

for i in range(len(all_case)):
    for j in range(EPISODE):
        if len(all_case[i][j])==3:
            vectors[i][j] = (node2vec[icd_netnode[all_case[i][j][0]]],
                         node2vec[icd_netnode[all_case[i][j][1]]],
                         node2vec[icd_netnode[all_case[i][j][2]]])
        if len(all_case[i][j])==2:
            vectors[i][j] = (node2vec[icd_netnode[all_case[i][j][0]]],
                         node2vec[icd_netnode[all_case[i][j][1]]],[0]*NET_EMBED_DIM)
        if len(all_case[i][j])==1:
            vectors[i][j] = (node2vec[icd_netnode[all_case[i][j][0]]],[0]*NET_EMBED_DIM,[0]*NET_EMBED_DIM)
        if len(all_case[i][j])==0:
            vectors[i][j] = ([0]*NET_EMBED_DIM, [0]*NET_EMBED_DIM, [0]*NET_EMBED_DIM)

#===transfer vectors[patient_indx][episode] = a 64*3 matrix
#===to vectors[patient_indx][episode] = a lsit, where 
#===the elements in the list is the absmax            
newvec=[]        
for patient_index in range(len(vectors)):
    for episode in range(EPISODE):
        for x, y, z in zip(vectors[patient_index][episode][0], vectors[patient_index][episode][1], vectors[patient_index][episode][2]):
            absmax = max( float(x), float(y), float(z), key=abs )
            newvec.append(absmax)
        vectors[patient_index][episode]=newvec
        newvec=[]
f = open('cases.csv','w')
for patient_index in range(len(vectors)):
    for episode in range(EPISODE):
        f.write(str(patient_index)+','+str(episode)+','+str(vectors[patient_index][episode])[1:-1]+'\n')
f.close()

#============CONTROLS===========
#===============================
#===============================
controls = pd.read_csv('C:/DATA/lowrisk_patients_records.csv',na_filter=False)
controls = controls[controls['AGE']!='nnnulll']
controls['AGE'] = controls['AGE'].astype(float)
controls = controls[controls['AGE']>10]
print (controls.shape[0])

d1 = controls[controls['diag_cd_01']!='null']
d2 = controls[controls['diag_cd_02']!='null']
d3 = controls[controls['diag_cd_03']!='null']
d4 = controls[controls['diag_cd_04']!='null']
d5 = controls[controls['diag_cd_05']!='null']
d6 = controls[controls['diag_cd_06']!='null']
d7 = controls[controls['diag_cd_07']!='null']
d8 = controls[controls['diag_cd_08']!='null']
d9 = controls[controls['diag_cd_09']!='null']
d10 = controls[controls['diag_cd_10']!='null']
(d1.shape[0]+d2.shape[0]+d3.shape[0]+d4.shape[0]+d5.shape[0]+d6.shape[0]+d7.shape[0]+d8.shape[0]+d9.shape[0]+d10.shape[0])/122114

individuals = list(set(list(controls['PATID'])))
controls_diaglist, sexlist, agelist, pidlist = [], [], [], []
for i in individuals[:]:
    records = controls[controls['PATID'] == i]
    if records.shape[0] >= 1: ##Patients without historical diagnoses will be removed
        agelist.append(records.iloc[0]['AGE'])
        sexlist.append(records.iloc[0]['SEX'])
        pidlist.append(records.iloc[0]['PATID'])           
len(list(set(pidlist)))  # 46460 patients, 136039 records        
f=open('age_controls','w')
for i in agelist:
    f.write(str(i)+'\n')
f.close()
f=open('sex_controls','w')
for i in sexlist:
    if i=="b'F'":
        f.write('0'+'\n')
    else:
        f.write('1'+'\n')
f.close()

#========
all_control = []
individuals =list(set(list( pidlist)))
for i in individuals[:]:
    ep = dict()
    for k in range(EPISODE): ep[k] = []

    records = controls[controls['PATID'] == i]
    records = records.sort_values(axis = 0, ascending = False, by = 'since2007')
    records.index = range(records.shape[0])
    records = records.drop(0) ##drop first line, i.e., sf record
    timeline = set(list(records['since2007']))
    
    count1 = 0 #count the # of episode
    for t in timeline:
        diags_in_an_episode = records[records['since2007'] == t]
        diags_in_an_episode = diags_in_an_episode[['diag_cd_01','diag_cd_02','diag_cd_03','diag_cd_04','diag_cd_05','diag_cd_06']].values
        diags_in_an_episode = list(diags_in_an_episode[0,:])
        diags_in_an_episode = list(filter(lambda x : x != 'null', diags_in_an_episode))  
        
        count2=0 #count the # of diagnoses in an episode
        for j in diags_in_an_episode:
            ep[count1].append(j)
            if count2>=2: break
            count2=count2+1    
        if count1>=2: break
        count1 = count1+1
    all_control.append(ep)
 
#==FIRST THREE DIGIT
temp2= []
for i in all_control:
    for j in i.keys():
        for d in i[j]:
            l = d.strip().split('.')
            temp2.append(l[0])
        i[j] = temp2
        temp2=[]

#==for nodes not in the network, assign a zero vector 
for i in range(1500):
    if str(i) not in node2vec.keys():
        node2vec[str(i)] = [0]*NET_EMBED_DIM

#==vectors[patient_indx][episode] = a 64*3 matrix
#all_case[patient_indx][episode][main/second/third_diagnosis]
m,n = len(list(set(pidlist))),3
vectors = [[0 for i in range(n)] for j in range(m)]

for i in range(len(all_control)):
    for j in range(EPISODE):
        if len(all_control[i][j])==3:
            vectors[i][j] = (node2vec[icd_netnode[all_control[i][j][0]]],
                         node2vec[icd_netnode[all_control[i][j][1]]],
                         node2vec[icd_netnode[all_control[i][j][2]]])
        if len(all_control[i][j])==2:
            vectors[i][j] = (node2vec[icd_netnode[all_control[i][j][0]]],
                         node2vec[icd_netnode[all_control[i][j][1]]],[0]*NET_EMBED_DIM)
        if len(all_control[i][j])==1:
            vectors[i][j] = (node2vec[icd_netnode[all_control[i][j][0]]],[0]*NET_EMBED_DIM,[0]*NET_EMBED_DIM)
        if len(all_control[i][j])==0:
            vectors[i][j] = ([0]*NET_EMBED_DIM, [0]*NET_EMBED_DIM, [0]*NET_EMBED_DIM)

#===transfer vectors[patient_indx][episode] = a 64*3 matrix
#===to vectors[patient_indx][episode] = a lsit, where 
#===the elements in the list is the absmax            
newvec=[]        
for patient_index in range(len(vectors)):
    for episode in range(EPISODE):
        for x, y, z in zip(vectors[patient_index][episode][0], vectors[patient_index][episode][1], vectors[patient_index][episode][2]):
            absmax = max( float(x), float(y), float(z), key=abs )
            newvec.append(absmax)
        vectors[patient_index][episode]=newvec
        newvec=[]

f = open('controls.csv','w')
for patient_index in range(len(vectors)):
    for episode in range(EPISODE):
        f.write(str(patient_index)+','+str(episode)+','+str(vectors[patient_index][episode])[1:-1]+'\n')
f.close()
               
#==========multi-hot==
#==========multi-hot==
diag_set=[]
patient_diag = dict()
for i in range(len(all_case)):
    patient_diag[i]=[]
    for j in range(EPISODE):
        if len(all_case[i][j])==3:
            patient_diag[i].append(all_case[i][j][0])
            patient_diag[i].append(all_case[i][j][1])
            patient_diag[i].append(all_case[i][j][2])
            diag_set.append(all_case[i][j][0])
            diag_set.append(all_case[i][j][1])
            diag_set.append(all_case[i][j][2])
        if len(all_case[i][j])==2:
            patient_diag[i].append(all_case[i][j][0])
            patient_diag[i].append(all_case[i][j][1])
            diag_set.append(all_case[i][j][0])
            diag_set.append(all_case[i][j][1])
        if len(all_case[i][j])==1:
            patient_diag[i].append(all_case[i][j][0])
            diag_set.append(all_case[i][j][0])

lowrisk_diag = dict()
for i in range(len(all_control)):
    lowrisk_diag[i]=[]
    for j in range(EPISODE):
        if len(all_control[i][j])==3:
            lowrisk_diag[i].append(all_control[i][j][0])
            lowrisk_diag[i].append(all_control[i][j][1])
            lowrisk_diag[i].append(all_control[i][j][2])
            diag_set.append(all_control[i][j][0])
            diag_set.append(all_control[i][j][1])
            diag_set.append(all_control[i][j][2])
        if len(all_control[i][j])==2:
            lowrisk_diag[i].append(all_control[i][j][0])
            lowrisk_diag[i].append(all_control[i][j][1])
            diag_set.append(all_control[i][j][0])
            diag_set.append(all_control[i][j][1])
        if len(all_control[i][j])==1:
            lowrisk_diag[i].append(all_control[i][j][0])
            diag_set.append(all_control[i][j][0])
diag_set = list(set(diag_set))


multihot = [0 for k in range( len(diag_set))]
icd_indx = dict()
for i in range(len(diag_set)):
    icd_indx[diag_set[i]] = i 
#==multihot_cases
f = open('multihot_cases.csv','w')
for i in patient_diag.keys():
    if len(patient_diag[i])>0:
        for j in patient_diag[i]:
            multihot[icd_indx[j]] = multihot[icd_indx[j]]+1
    else: 
        multihot = multihot
    f.write(str(i)+','+str(multihot)[1:-1]+'\n')
    multihot = [0 for k in range( len(diag_set))]
f.close()
#==multihot_controls
f = open('multihot_controls.csv','w')
for i in lowrisk_diag.keys():
    if len(lowrisk_diag[i])>0:
        for j in lowrisk_diag[i]:
            multihot[icd_indx[j]] = multihot[icd_indx[j]]+1
    else: 
        multihot = multihot
    f.write(str(i)+','+str(multihot)[1:-1]+'\n')
    multihot = [0 for k in range( len(diag_set))]
f.close()

##======sigle diag
# emb
single_diag = pd.DataFrame()
single_diag['diag'] = diag_set
emb =[]
for i in diag_set:
    emb.append(node2vec[icd_netnode[i]])
    
for i in range(NET_EMBED_DIM):
    single_diag['emb_'+str(i)] = 0
single_diag.iloc[:,1:]=emb
for i in range(NET_EMBED_DIM, NET_EMBED_DIM*EPISODE):
    single_diag['emb_'+str(i)] = 0

# multihot_singles
multihot = [0 for k in range( len(diag_set))]
mhot = []
for i in diag_set:
    multihot[icd_indx[i]] = multihot[icd_indx[i]]+1
    mhot.append(multihot)
    multihot = [0 for k in range( len(diag_set))]
for i in range(len(diag_set)):
    single_diag['mhot_'+str(i)] = 0
single_diag.iloc[:,1+NET_EMBED_DIM*EPISODE:] = mhot
single_diag['AGE'] = 50
single_diag['SEX'] = 0
single_diag['y'] = 2

##Please use "single_diag" directly in prediction.py



