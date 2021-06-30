#!/usr/bin/env python
# coding: utf-8


#%%
pip install numpy==1.19.3 --user
pip install tensorflow --user

#%%

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter


# In[ ]:


df=pd.read_csv('/Users/pau26/OneDrive/Documentos/BEDU/Machine Learning/Proyecto final/Reportes.csv') #cargar archivo
df=df[df.isnull().DESCRIPTION==False] #quitar los casos que no tengan descripcion


# In[ ]:


cat=np.array(df['CATEGORY'])

cat1=[]
cat2=[]

for i in cat:
    cat2.append([])
    for j in i[:-2].split(','):
        k=j.strip()
        cat2[-1].append(k.lower())
        if k.lower() not in cat1:
            cat1.append(k.lower())



# In[ ]:


column_names=np.unique(cat1)
row_names=np.array(range(len(cat)))

matrix = np.zeros((len(row_names),len(column_names))).astype('int') #crear matriz de ceros
dum = pd.DataFrame(matrix, columns=column_names, index=row_names) #hacer DF cuyas columnas se llamen como las categorias de agresion
#y las filas esten reenumeradas porque la numeracion original estaba mal

for i in row_names: #se marca con 1 si la descripcion corresponde al tipo de agresion en la columna correspondiente. Si no, se queda en 0
    for j in cat2[i]:
        dum[j][i]=1
        
df['ind']=row_names #se agrega una nueva columna a la base original
df=df.set_index('ind') #se reasignan los indices de la base original

data=pd.concat([df, dum], axis=1) #se unen la base original y la base "dum"


# In[ ]:


des=df['DESCRIPTION']

v=[]

for i in range(len(des)): #juntar todas las palabras de todas las denuncias salvo los simbolos de "replace" e irlos guardando en "v"
    a=des[i]    
    a=a.replace("."," ")
    a=a.replace("\n","")
    a=a.replace("!","")
    a=a.replace("#","")
    a=a.replace(";"," ")
    a=a.replace("'","")
    a=a.replace('"','')
    a=a.replace("(","")
    a=a.replace(")","")
    a=a.replace(",","")
    a=a.replace("&"," ")
    a=a.replace("quot","")
    
    v=np.concatenate((v,np.unique(a.split(' '))))
    
cuentas=Counter(v) #contar las veces que aparece cada palabra en "v"

diccionario={}
voc1=[]

for i in cuentas: #guardar solo las palabras que aparezcan mas de 5 veces (no interesan nombres o telefonos)
    if cuentas[i]>5:
        voc1.append(i)
        
diccionario = dict(zip(voc1, range(len(voc1)))) #hacer un diccionario


# In[ ]:


desc=[]
A=np.array(data['DESCRIPTION']) #todas las descripciones

for i in range(len(A)): #hacer la misma discriminacion de simbolos que al crear el diccionario
    desc.append([])
    a=A[i]
    a=a.replace("."," ")
    a=a.replace("\n","")
    a=a.replace("!","")
    a=a.replace("#","")
    a=a.replace(";"," ")
    a=a.replace("'","")
    a=a.replace('"','')
    a=a.replace("(","")
    a=a.replace(")","")
    a=a.replace(",","")
    a=a.replace("&"," ")
    a=a.replace("quot","")
    
    for j in a.split(' '): #si la palabra aparece en el diccionario, asignarle el numero que le toca
        if j.strip().lower() in diccionario.keys():
            desc[-1].append(diccionario[j.strip().lower()])
        else: #si la palabra no aparece en el diccionario, asignarle -1
            desc[-1].append(-1)


# In[ ]:


M=np.zeros((len(np.array(df['DESCRIPTION'])),len(diccionario))) #matriz de ceros de tamaño (cantidad de denuncias)x(longitud de diccionario)

for i in range(len(M)): #asignarle a cada denuncia la cantidad de veces que aparece cada palabra
    for j in desc[i]:
        M[i][int(j)]=M[i][int(j)]+1


# In[ ]:


X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario))) #los datos de entrada seran las cuentas guardadas en M
Y=np.transpose(np.transpose(data.values)[3:]) #las etiquetas seran la clasificacion binaria que hicimos en "dum"
Y1=np.transpose(Y)[0] #tomamos las etiquetas para el primer tipo de denuncia

X_train=np.array(X[:int(len(X)*0.9)]) #entrenamos con un 90% de los datos
Y_train=np.array(Y1[:int(len(X)*0.9)])

model_0 = tf.keras.Sequential() #modelo
model_0.add(tf.keras.layers.Dense(len(X_train[0][0]), activation='relu'))
model_0.add(tf.keras.layers.Dense(int(len(X_train[0][0])*0.5), activation='tanh'))
model_0.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model_0.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train=np.asarray(X_train).astype(np.float32) #convertir los datos en flotantes
Y_train=np.asarray(Y_train).astype(np.float32)

model_0.fit(X_train,Y_train, epochs=7) #entrenar el modelo

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario))) #los datos de entrada seran las cuentas guardadas en M
Y=np.transpose(np.transpose(data.values)[3:]) #las etiquetas seran la clasificacion binaria que hicimos en "dum"
Y1=np.transpose(Y)[1] #tomamos las etiquetas para el primer tipo de denuncia

X_train=np.array(X[:int(len(X)*0.9)]) #entrenamos con un 90% de los datos
Y_train=np.array(Y1[:int(len(X)*0.9)])

model_1 = tf.keras.Sequential() #modelo
model_1.add(tf.keras.layers.Dense(len(X_train[0][0]), activation='relu'))
model_1.add(tf.keras.layers.Dense(int(len(X_train[0][0])*0.5), activation='tanh'))
model_1.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train=np.asarray(X_train).astype(np.float32) #convertir los datos en flotantes
Y_train=np.asarray(Y_train).astype(np.float32)

model_1.fit(X_train,Y_train, epochs=7) #entrenar el modelo

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario))) #los datos de entrada seran las cuentas guardadas en M
Y=np.transpose(np.transpose(data.values)[3:]) #las etiquetas seran la clasificacion binaria que hicimos en "dum"
Y1=np.transpose(Y)[2] #tomamos las etiquetas para el primer tipo de denuncia

X_train=np.array(X[:int(len(X)*0.9)]) #entrenamos con un 90% de los datos
Y_train=np.array(Y1[:int(len(X)*0.9)])

model_2 = tf.keras.Sequential() #modelo
model_2.add(tf.keras.layers.Dense(len(X_train[0][0]), activation='relu'))
model_2.add(tf.keras.layers.Dense(int(len(X_train[0][0])*0.5), activation='tanh'))
model_2.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train=np.asarray(X_train).astype(np.float32) #convertir los datos en flotantes
Y_train=np.asarray(Y_train).astype(np.float32)

model_2.fit(X_train,Y_train, epochs=7) #entrenar el modelo

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario))) #los datos de entrada seran las cuentas guardadas en M
Y=np.transpose(np.transpose(data.values)[3:]) #las etiquetas seran la clasificacion binaria que hicimos en "dum"
Y1=np.transpose(Y)[3] #tomamos las etiquetas para el primer tipo de denuncia

X_train=np.array(X[:int(len(X)*0.9)]) #entrenamos con un 90% de los datos
Y_train=np.array(Y1[:int(len(X)*0.9)])

model_3 = tf.keras.Sequential() #modelo
model_3.add(tf.keras.layers.Dense(len(X_train[0][0]), activation='relu'))
model_3.add(tf.keras.layers.Dense(int(len(X_train[0][0])*0.5), activation='tanh'))
model_3.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model_3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train=np.asarray(X_train).astype(np.float32) #convertir los datos en flotantes
Y_train=np.asarray(Y_train).astype(np.float32)

model_3.fit(X_train,Y_train, epochs=7) #entrenar el modelo

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario))) #los datos de entrada seran las cuentas guardadas en M
Y=np.transpose(np.transpose(data.values)[3:]) #las etiquetas seran la clasificacion binaria que hicimos en "dum"
Y1=np.transpose(Y)[4] #tomamos las etiquetas para el primer tipo de denuncia

X_train=np.array(X[:int(len(X)*0.9)]) #entrenamos con un 90% de los datos
Y_train=np.array(Y1[:int(len(X)*0.9)])

model_4 = tf.keras.Sequential() #modelo
model_4.add(tf.keras.layers.Dense(len(X_train[0][0]), activation='relu'))
model_4.add(tf.keras.layers.Dense(int(len(X_train[0][0])*0.5), activation='tanh'))
model_4.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model_4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train=np.asarray(X_train).astype(np.float32) #convertir los datos en flotantes
Y_train=np.asarray(Y_train).astype(np.float32)

model_4.fit(X_train,Y_train, epochs=7) #entrenar el modelo

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario))) #los datos de entrada seran las cuentas guardadas en M
Y=np.transpose(np.transpose(data.values)[3:]) #las etiquetas seran la clasificacion binaria que hicimos en "dum"
Y1=np.transpose(Y)[5] #tomamos las etiquetas para el primer tipo de denuncia

X_train=np.array(X[:int(len(X)*0.9)]) #entrenamos con un 90% de los datos
Y_train=np.array(Y1[:int(len(X)*0.9)])

model_5 = tf.keras.Sequential() #modelo
model_5.add(tf.keras.layers.Dense(len(X_train[0][0]), activation='relu'))
model_5.add(tf.keras.layers.Dense(int(len(X_train[0][0])*0.5), activation='tanh'))
model_5.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model_5.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train=np.asarray(X_train).astype(np.float32) #convertir los datos en flotantes
Y_train=np.asarray(Y_train).astype(np.float32)

model_5.fit(X_train,Y_train, epochs=4) #entrenar el modelo

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario))) #los datos de entrada seran las cuentas guardadas en M
Y=np.transpose(np.transpose(data.values)[3:]) #las etiquetas seran la clasificacion binaria que hicimos en "dum"
Y1=np.transpose(Y)[6] #tomamos las etiquetas para el primer tipo de denuncia

X_train=np.array(X[:int(len(X)*0.9)]) #entrenamos con un 90% de los datos
Y_train=np.array(Y1[:int(len(X)*0.9)])

model_6 = tf.keras.Sequential() #modelo
model_6.add(tf.keras.layers.Dense(len(X_train[0][0]), activation='relu'))
model_6.add(tf.keras.layers.Dense(int(len(X_train[0][0])*0.5), activation='tanh'))
model_6.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model_6.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train=np.asarray(X_train).astype(np.float32) #convertir los datos en flotantes
Y_train=np.asarray(Y_train).astype(np.float32)

model_6.fit(X_train,Y_train, epochs=4) #entrenar el modelo

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario))) #los datos de entrada seran las cuentas guardadas en M
Y=np.transpose(np.transpose(data.values)[3:]) #las etiquetas seran la clasificacion binaria que hicimos en "dum"
Y1=np.transpose(Y)[7] #tomamos las etiquetas para el primer tipo de denuncia

X_train=np.array(X[:int(len(X)*0.9)]) #entrenamos con un 90% de los datos
Y_train=np.array(Y1[:int(len(X)*0.9)])

model_7 = tf.keras.Sequential() #modelo
model_7.add(tf.keras.layers.Dense(len(X_train[0][0]), activation='relu'))
model_7.add(tf.keras.layers.Dense(int(len(X_train[0][0])*0.5), activation='tanh'))
model_7.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model_7.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train=np.asarray(X_train).astype(np.float32) #convertir los datos en flotantes
Y_train=np.asarray(Y_train).astype(np.float32)

model_7.fit(X_train,Y_train, epochs=4) #entrenar el modelo

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario))) #los datos de entrada seran las cuentas guardadas en M
Y=np.transpose(np.transpose(data.values)[3:]) #las etiquetas seran la clasificacion binaria que hicimos en "dum"
Y1=np.transpose(Y)[8] #tomamos las etiquetas para el primer tipo de denuncia

X_train=np.array(X[:int(len(X)*0.9)]) #entrenamos con un 90% de los datos
Y_train=np.array(Y1[:int(len(X)*0.9)])

model_8 = tf.keras.Sequential() #modelo
model_8.add(tf.keras.layers.Dense(len(X_train[0][0]), activation='relu'))
model_8.add(tf.keras.layers.Dense(int(len(X_train[0][0])*0.5), activation='tanh'))
model_8.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model_8.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train=np.asarray(X_train).astype(np.float32) #convertir los datos en flotantes
Y_train=np.asarray(Y_train).astype(np.float32)

model_8.fit(X_train,Y_train, epochs=4) #entrenar el modelo

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario))) #los datos de entrada seran las cuentas guardadas en M
Y=np.transpose(np.transpose(data.values)[3:]) #las etiquetas seran la clasificacion binaria que hicimos en "dum"
Y1=np.transpose(Y)[9] #tomamos las etiquetas para el primer tipo de denuncia

X_train=np.array(X[:int(len(X)*0.9)]) #entrenamos con un 90% de los datos
Y_train=np.array(Y1[:int(len(X)*0.9)])

model_9 = tf.keras.Sequential() #modelo
model_9.add(tf.keras.layers.Dense(len(X_train[0][0]), activation='relu'))
model_9.add(tf.keras.layers.Dense(int(len(X_train[0][0])*0.5), activation='tanh'))
model_9.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model_9.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train=np.asarray(X_train).astype(np.float32) #convertir los datos en flotantes
Y_train=np.asarray(Y_train).astype(np.float32)

model_9.fit(X_train,Y_train, epochs=4) #entrenar el modelo

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario))) #los datos de entrada seran las cuentas guardadas en M
Y=np.transpose(np.transpose(data.values)[3:]) #las etiquetas seran la clasificacion binaria que hicimos en "dum"
Y1=np.transpose(Y)[10] #tomamos las etiquetas para el primer tipo de denuncia

X_train=np.array(X[:int(len(X)*0.9)]) #entrenamos con un 90% de los datos
Y_train=np.array(Y1[:int(len(X)*0.9)])

model_10 = tf.keras.Sequential() #modelo
model_10.add(tf.keras.layers.Dense(len(X_train[0][0]), activation='relu'))
model_10.add(tf.keras.layers.Dense(int(len(X_train[0][0])*0.5), activation='tanh'))
model_10.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model_10.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train=np.asarray(X_train).astype(np.float32) #convertir los datos en flotantes
Y_train=np.asarray(Y_train).astype(np.float32)

model_10.fit(X_train,Y_train, epochs=4) #entrenar el modelo

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario))) #los datos de entrada seran las cuentas guardadas en M
Y=np.transpose(np.transpose(data.values)[3:]) #las etiquetas seran la clasificacion binaria que hicimos en "dum"
Y1=np.transpose(Y)[11] #tomamos las etiquetas para el primer tipo de denuncia

X_train=np.array(X[:int(len(X)*0.9)]) #entrenamos con un 90% de los datos
Y_train=np.array(Y1[:int(len(X)*0.9)])

model_11 = tf.keras.Sequential() #modelo
model_11.add(tf.keras.layers.Dense(len(X_train[0][0]), activation='relu'))
model_11.add(tf.keras.layers.Dense(int(len(X_train[0][0])*0.5), activation='tanh'))
model_11.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model_11.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train=np.asarray(X_train).astype(np.float32) #convertir los datos en flotantes
Y_train=np.asarray(Y_train).astype(np.float32)

model_11.fit(X_train,Y_train, epochs=4) #entrenar el modelo

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario))) #los datos de entrada seran las cuentas guardadas en M
Y=np.transpose(np.transpose(data.values)[3:]) #las etiquetas seran la clasificacion binaria que hicimos en "dum"
Y1=np.transpose(Y)[12] #tomamos las etiquetas para el primer tipo de denuncia

X_train=np.array(X[:int(len(X)*0.9)]) #entrenamos con un 90% de los datos
Y_train=np.array(Y1[:int(len(X)*0.9)])

model_12 = tf.keras.Sequential() #modelo
model_12.add(tf.keras.layers.Dense(len(X_train[0][0]), activation='relu'))
model_12.add(tf.keras.layers.Dense(int(len(X_train[0][0])*0.5), activation='tanh'))
model_12.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model_12.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train=np.asarray(X_train).astype(np.float32) #convertir los datos en flotantes
Y_train=np.asarray(Y_train).astype(np.float32)

model_12.fit(X_train,Y_train, epochs=4) #entrenar el modelo

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario))) #los datos de entrada seran las cuentas guardadas en M
Y=np.transpose(np.transpose(data.values)[3:]) #las etiquetas seran la clasificacion binaria que hicimos en "dum"
Y1=np.transpose(Y)[13] #tomamos las etiquetas para el primer tipo de denuncia

X_train=np.array(X[:int(len(X)*0.9)]) #entrenamos con un 90% de los datos
Y_train=np.array(Y1[:int(len(X)*0.9)])

model_13 = tf.keras.Sequential() #modelo
model_13.add(tf.keras.layers.Dense(len(X_train[0][0]), activation='relu'))
model_13.add(tf.keras.layers.Dense(int(len(X_train[0][0])*0.5), activation='tanh'))
model_13.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model_13.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train=np.asarray(X_train).astype(np.float32) #convertir los datos en flotantes
Y_train=np.asarray(Y_train).astype(np.float32)

model_13.fit(X_train,Y_train, epochs=4) #entrenar el modelo


#%%

#Guardar todas las neuronas que pesan mucho,
#model_0.save('/Users/pau26/OneDrive/Documentos/BEDU/Machine Learning/Proyecto final/modelo_0.h5')
#model_1.save('/Users/pau26/OneDrive/Documentos/BEDU/Machine Learning/Proyecto final/modelo_1.h5')
#model_2.save('/Users/pau26/OneDrive/Documentos/BEDU/Machine Learning/Proyecto final/modelo_2.h5')
#model_3.save('/Users/pau26/OneDrive/Documentos/BEDU/Machine Learning/Proyecto final/modelo_3.h5')
#model_4.save('/Users/pau26/OneDrive/Documentos/BEDU/Machine Learning/Proyecto final/modelo_4.h5')
#model_5.save('/Users/pau26/OneDrive/Documentos/BEDU/Machine Learning/Proyecto final/modelo_5.h5')
#model_6.save('/Users/pau26/OneDrive/Documentos/BEDU/Machine Learning/Proyecto final/modelo_6.h5')
#model_7.save('/Users/pau26/OneDrive/Documentos/BEDU/Machine Learning/Proyecto final/modelo_7.h5')
#model_8.save('/Users/pau26/OneDrive/Documentos/BEDU/Machine Learning/Proyecto final/modelo_8.h5')
#model_9.save('/Users/pau26/OneDrive/Documentos/BEDU/Machine Learning/Proyecto final/modelo_9.h5') 
#model_10.save('/Users/pau26/OneDrive/Documentos/BEDU/Machine Learning/Proyecto final/modelo_10.h5')
#model_11.save('/Users/pau26/OneDrive/Documentos/BEDU/Machine Learning/Proyecto final/modelo_11.h5')
#model_12.save('/Users/pau26/OneDrive/Documentos/BEDU/Machine Learning/Proyecto final/modelo_12.h5')
#model_13.save('/Users/pau26/OneDrive/Documentos/BEDU/Machine Learning/Proyecto final/modelo_13.h5')


#%%
#Probar los modelos con la base de la India
C=[]

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data.values)[3:])

X_test=np.array(X[int(len(X)*0.9):])
Y_test=np.array(Y[int(len(X)*0.9):])
Y1=np.transpose(Y_test)[0]
c=0

for i in range(len(Y1)):
    if np.round(model_0.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))
C
#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data.values)[3:])

X_test=np.array(X[int(len(X)*0.9):])
Y_test=np.array(Y[int(len(X)*0.9):])
Y1=np.transpose(Y_test)[1]
c=0

for i in range(len(Y1)):
    if np.round(model_1.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

C

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data.values)[3:])

X_test=np.array(X[int(len(X)*0.9):])
Y_test=np.array(Y[int(len(X)*0.9):])
Y1=np.transpose(Y_test)[2]
c=0

for i in range(len(Y1)):
    if np.round(model_2.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

C

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data.values)[3:])

X_test=np.array(X[int(len(X)*0.9):])
Y_test=np.array(Y[int(len(X)*0.9):])
Y1=np.transpose(Y_test)[3]
c=0

for i in range(len(Y1)):
    if np.round(model_3.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))


#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data.values)[3:])

X_test=np.array(X[int(len(X)*0.9):])
Y_test=np.array(Y[int(len(X)*0.9):])
Y1=np.transpose(Y_test)[4]
c=0

for i in range(len(Y1)):
    if np.round(model_4.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

C

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data.values)[3:])

X_test=np.array(X[int(len(X)*0.9):])
Y_test=np.array(Y[int(len(X)*0.9):])
Y1=np.transpose(Y_test)[5]
c=0

for i in range(len(Y1)):
    if np.round(model_5.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data.values)[3:])

X_test=np.array(X[int(len(X)*0.9):])
Y_test=np.array(Y[int(len(X)*0.9):])
Y1=np.transpose(Y_test)[6]
c=0

for i in range(len(Y1)):
    if np.round(model_6.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data.values)[3:])

X_test=np.array(X[int(len(X)*0.9):])
Y_test=np.array(Y[int(len(X)*0.9):])
Y1=np.transpose(Y_test)[7]
c=0

for i in range(len(Y1)):
    if np.round(model_7.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data.values)[3:])

X_test=np.array(X[int(len(X)*0.9):])
Y_test=np.array(Y[int(len(X)*0.9):])
Y1=np.transpose(Y_test)[8]
c=0

for i in range(len(Y1)):
    if np.round(model_8.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data.values)[3:])

X_test=np.array(X[int(len(X)*0.9):])
Y_test=np.array(Y[int(len(X)*0.9):])
Y1=np.transpose(Y_test)[9]
c=0

for i in range(len(Y1)):
    if np.round(model_9.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data.values)[3:])

X_test=np.array(X[int(len(X)*0.9):])
Y_test=np.array(Y[int(len(X)*0.9):])
Y1=np.transpose(Y_test)[10]
c=0

for i in range(len(Y1)):
    if np.round(model_10.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data.values)[3:])

X_test=np.array(X[int(len(X)*0.9):])
Y_test=np.array(Y[int(len(X)*0.9):])
Y1=np.transpose(Y_test)[11]
c=0

for i in range(len(Y1)):
    if np.round(model_11.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data.values)[3:])

X_test=np.array(X[int(len(X)*0.9):])
Y_test=np.array(Y[int(len(X)*0.9):])
Y1=np.transpose(Y_test)[12]
c=0

for i in range(len(Y1)):
    if np.round(model_12.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

#%%

X=np.reshape(M,(len(np.array(df['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data.values)[3:])

X_test=np.array(X[int(len(X)*0.9):])
Y_test=np.array(Y[int(len(X)*0.9):])
Y1=np.transpose(Y_test)[13]
c=0

for i in range(len(Y1)):
    if np.round(model_13.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

#%%
# PORCENTAJE DE ACIERTO BASE DE LA INDIA
C
acierto = sum(C)/len(C)
acierto

############################################################ AQUI EMPIEZA IPN

#%%

df_IPN=pd.read_csv('/Users/pau26/OneDrive/Documentos/BEDU/Machine Learning/Proyecto final/Base_IPN.csv', encoding='latin1')

#%%
cat_IPN=np.array(df_IPN['CATEGORY'])

cat1_IPN=[]
cat2_IPN=[]
cont=0
for i in cat_IPN:
    cat2_IPN.append([])
    cont+=1
    #print(f'i {cont} {i}') 
    for j in i.split(','):
        k=j.strip()
        cat2_IPN[-1].append(k.lower())
        #print('j'+j)
        if k.lower() not in cat1_IPN:
            cat1_IPN.append(k.lower())
            

#%%
column_names_IPN=np.unique(cat1)
row_names_IPN=np.array(range(len(cat_IPN)))

matrix_IPN = np.zeros((len(row_names_IPN),len(column_names_IPN))).astype('int') #crear matriz de ceros
dum_IPN = pd.DataFrame(matrix_IPN, columns=column_names_IPN, index=row_names_IPN)

cont=0
cont1=0
for i in row_names_IPN: #se marca con 1 si la descripcion corresponde al tipo de agresion en la columna correspondiente. Si no, se queda en 0
    #print(f'i {cont} {i}') 
    cont+=1
    for j in cat2_IPN[i]:
        #print(f'j {cont1} {j}') 
        dum_IPN[j][i]=1
        cont1+=1
        
df_IPN['ind']=row_names_IPN #se agrega una nueva columna a la base original
df_IPN=df_IPN.set_index('ind') #se reasignan los indices de la base original

data_IPN=pd.concat([df_IPN, dum_IPN], axis=1) #se unen la base original y la base "dum"


#%% 

desc_IPN=[]
A_IPN=np.array(df_IPN['DESCRIPTION']) #todas las descripciones

for i in range(len(A_IPN)): #hacer la misma discriminacion de simbolos que al crear el diccionario
    desc_IPN.append([])
    a_IPN=A_IPN[i]
    a_IPN=a_IPN.replace("."," ")
    a_IPN=a_IPN.replace("\n","")
    a_IPN=a_IPN.replace("!","")
    a_IPN=a_IPN.replace("#","")
    a_IPN=a_IPN.replace(";"," ")
    a_IPN=a_IPN.replace("'","")
    a_IPN=a_IPN.replace('"','')
    a_IPN=a_IPN.replace("(","")
    a_IPN=a_IPN.replace(")","")
    a_IPN=a_IPN.replace(",","")
    a_IPN=a_IPN.replace("&"," ")
    a_IPN=a_IPN.replace("quot","")
    
    for j in a_IPN.split(' '): #si la palabra aparece en el diccionario, asignarle el numero que le toca
        if j.strip().lower() in diccionario.keys():
            desc_IPN[-1].append(diccionario[j.strip().lower()])
        else: #si la palabra no aparece en el diccionario, asignarle -1
            desc_IPN[-1].append(-1)

#%%
M_test=np.zeros((len(np.array(df_IPN['DESCRIPTION'])),len(diccionario))) #matriz de ceros de tamaño (cantidad de denuncias)x(longitud de diccionario)

for i in range(len(M_test)): #asignarle a cada denuncia la cantidad de veces que aparece cada palabra
    for j in desc[i]:
        M_test[i][int(j)]=M_test[i][int(j)]+1

#%%

C=[]

X=np.reshape(M_test,(len(np.array(df_IPN['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data_IPN.values)[3:])

X_test=np.array(X[:int(len(X)*1)]) 
Y_test=np.array(Y[:int(len(X)*1)]) 
Y1=np.transpose(Y_test)[0]
c=0

for i in range(len(Y1)):
    if np.round(model_0.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))


C
#%% #Probando el modelo con la base del IPN

X=np.reshape(M_test,(len(np.array(df_IPN['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data_IPN.values)[3:])

X_test=np.array(X[:int(len(X)*.99)]) 
Y_test=np.array(Y[:int(len(X)*.99)]) 
Y1=np.transpose(Y_test)[1]
c=0

for i in range(len(Y1)):
    if np.round(model_1.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

#%%
X=np.reshape(M_test,(len(np.array(df_IPN['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data_IPN.values)[3:])

X_test=np.array(X[:int(len(X)*1)]) 
Y_test=np.array(Y[:int(len(X)*1)]) 
Y1=np.transpose(Y_test)[2]
c=0

for i in range(len(Y1)):
    if np.round(model_2.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

#%%
X=np.reshape(M_test,(len(np.array(df_IPN['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data_IPN.values)[3:])

X_test=np.array(X[:int(len(X)*1)]) 
Y_test=np.array(Y[:int(len(X)*1)]) 
Y1=np.transpose(Y_test)[3]
c=0

for i in range(len(Y1)):
    if np.round(model_3.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

#%%

X=np.reshape(M_test,(len(np.array(df_IPN['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data_IPN.values)[3:])

X_test=np.array(X[:int(len(X)*1)]) 
Y_test=np.array(Y[:int(len(X)*1)]) 
Y1=np.transpose(Y_test)[4]
c=0

for i in range(len(Y1)):
    if np.round(model_4.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

#%%
X=np.reshape(M_test,(len(np.array(df_IPN['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data_IPN.values)[3:])

X_test=np.array(X[:int(len(X)*1)]) 
Y_test=np.array(Y[:int(len(X)*1)]) 
Y1=np.transpose(Y_test)[5]
c=0

for i in range(len(Y1)):
    if np.round(model_5.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

#%%
X=np.reshape(M_test,(len(np.array(df_IPN['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data_IPN.values)[3:])

X_test=np.array(X[:int(len(X)*1)]) 
Y_test=np.array(Y[:int(len(X)*1)]) 
Y1=np.transpose(Y_test)[6]
c=0

for i in range(len(Y1)):
    if np.round(model_6.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

#%%
X=np.reshape(M_test,(len(np.array(df_IPN['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data_IPN.values)[3:])

X_test=np.array(X[:int(len(X)*1)]) 
Y_test=np.array(Y[:int(len(X)*1)]) 
Y1=np.transpose(Y_test)[7]
c=0

for i in range(len(Y1)):
    if np.round(model_7.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

#%%
X=np.reshape(M_test,(len(np.array(df_IPN['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data_IPN.values)[3:])

X_test=np.array(X[:int(len(X)*1)]) 
Y_test=np.array(Y[:int(len(X)*1)]) 
Y1=np.transpose(Y_test)[8]
c=0

for i in range(len(Y1)):
    if np.round(model_8.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

#%%

X=np.reshape(M_test,(len(np.array(df_IPN['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data_IPN.values)[3:])

X_test=np.array(X[:int(len(X)*1)]) 
Y_test=np.array(Y[:int(len(X)*1)]) 
Y1=np.transpose(Y_test)[9]
c=0

for i in range(len(Y1)):
    if np.round(model_9.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

#%%

X=np.reshape(M_test,(len(np.array(df_IPN['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data_IPN.values)[3:])

X_test=np.array(X[:int(len(X)*1)]) 
Y_test=np.array(Y[:int(len(X)*1)]) 
Y1=np.transpose(Y_test)[10]
c=0

for i in range(len(Y1)):
    if np.round(model_10.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

#%%

X=np.reshape(M_test,(len(np.array(df_IPN['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data_IPN.values)[3:])

X_test=np.array(X[:int(len(X)*1)]) 
Y_test=np.array(Y[:int(len(X)*1)]) 
Y1=np.transpose(Y_test)[11]
c=0

for i in range(len(Y1)):
    if np.round(model_11.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

#%%

X=np.reshape(M_test,(len(np.array(df_IPN['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data_IPN.values)[3:])

X_test=np.array(X[:int(len(X)*1)]) 
Y_test=np.array(Y[:int(len(X)*1)]) 
Y1=np.transpose(Y_test)[12]
c=0

for i in range(len(Y1)):
    if np.round(model_12.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))

#%%

X=np.reshape(M_test,(len(np.array(df_IPN['DESCRIPTION'])),1,len(diccionario)))
Y=np.transpose(np.transpose(data_IPN.values)[3:])

X_test=np.array(X[:int(len(X)*1)]) 
Y_test=np.array(Y[:int(len(X)*1)]) 
Y1=np.transpose(Y_test)[13]
c=0

for i in range(len(Y1)):
    if np.round(model_13.predict(X_test[i])[0][0])==Y1[i]:
        c+=1
        
C.append(c/len(Y1))
C

#%%
# PORCENTAJE DE ACIERTO BASE IPN
C
acierto_IPN = sum(C)/len(C)
acierto_IPN  #0.91

cat1
