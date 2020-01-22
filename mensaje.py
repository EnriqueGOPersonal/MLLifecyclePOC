# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 23:15:29 2020

@author: enriq
"""

import pandas as pd
import numpy as np

characters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r', "s", 't','u','v','w','x','y','z']
numbers = range(len(characters))

dic = {c: v for c, v in zip(characters, numbers)}
dic2 = {c: v for c, v in zip(numbers, characters)}

str_cod = "IWAW NZFRN JKWBME HZA URMI EYHMXSF HL RPUKVB HV XZTKTUSEI YM MPBFRRO I TIJ CFVF"
str_cod = list(str_cod.lower().replace(" ",""))
str_num = str_cod.copy()

df_num = []
for j, k in enumerate(str_cod): # Cada letra quiero sustituirla en decod con el valor que le diga del dic
    str_num[j] = dic[k]

for i in range(len(characters)):
    df_num.append(str_num)

df = pd.DataFrame(df_num)

for i in range(len(df)):
    df.loc[i, :] = df.loc[i, :] + i
    df.loc[i, :] = df.loc[i, :] % len(characters)
    df.loc[i, :] = [dic2[x] for x in df.loc[i, :]]

# ----------------------------------------------------- Mensaje en reversa
    
str_cod = "IWAW NZFRN JKWBME HZA URMI EYHMXSF HL RPUKVB HV XZTKTUSEI YM MPBFRRO I TIJ CFVF"
str_cod = ''.join(reversed(str_cod))
str_cod = list(str_cod.lower().replace(" ",""))
str_num = str_cod.copy()

df_num = []
for j, k in enumerate(str_cod): # Cada letra quiero sustituirla en decod con el valor que le diga del dic
    str_num[j] = dic[k]

for i in range(len(characters)):
    df_num.append(str_num)

df = pd.DataFrame(df_num)

for i in range(len(df)):
    df.loc[i, :] = df.loc[i, :] + i
    df.loc[i, :] = df.loc[i, :] % len(characters)
    df.loc[i, :] = [dic2[x] for x in df.loc[i, :]]

