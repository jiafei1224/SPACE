#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import re
import json

with open("./contact/train.json") as f:
    data = json.load(f)    
res = list(data.values())

sphere=0
spherecount=0
cone=0
conecount=0
cube=0
cubecount=0
cylinder=0
cylindercount=0
torus=0
toruscount=0
invertedcone=0
invertedconecount=0
sidedcylinder=0
sidedcylindercount=0
for i in range(len(res)):
    numofobject=len(res[i])
#     print(res[i])
    for b in range(len(res[i])):
        obj2=res[i][b][0]
        obj3=res[i][b][2]
        obj = re.sub(r'[^a-zA-Z ]+', '', obj2)
        if obj == 'Sphere':
            sphere = sphere +1
            if str(obj3) == '[1]':
                spherecount=spherecount+1
                
        elif obj == 'Cone':
            cone=cone +1
            if str(obj3) == '[1]':
                conecount=conecount+1
                
        elif obj == 'Cube':
            cube=cube +1
            if str(obj3) == '[1]':
                cubecount=cubecount+1
                
        elif obj == 'Cylinder':
            cylinder=cylinder +1
            if str(obj3) == '[1]':
                cylindercount=cylindercount+1
                
        elif obj == 'Torus':
            torus=torus +1
            if str(obj3) == '[1]':
                toruscount=toruscount+1
                
        elif obj == 'InvertedCone':
            invertedcone=invertedcone +1
            if str(obj3) == '[1]':
                invertedconecount=invertedconecount+1
                
        elif obj == 'SideCylinder':
            sidedcylinder=sidedcylinder +1
            if str(obj3) == '[1]':
                sidedcylindercount=sidedcylindercount+1
                
print("Sphere:"+ str(sphere))
print("Sphere count:"+ str(spherecount))

print("Cone:"+ str(cone))
print("Cone count:"+ str(conecount))

print("Cube:"+ str(cube))
print("Cube count:"+ str(cubecount))

print("Cylinder:"+ str(cylinder))
print("Cylinder count:"+ str(cylindercount))

print("Torus:"+ str(torus))
print("Torus count:"+ str(toruscount))

print("InvertedCone:"+ str(invertedcone))
print("InvertedCone count:"+ str(invertedconecount))

print("SideCylinder:"+ str(sidedcylinder))
print("SideCylinder count:"+ str(sidedcylindercount))


# In[2]:


import csv
import re
import json

with open("./contact/train.json") as f:
    data = json.load(f)    
res = list(data.values())
print(res)


# In[3]:


import csv
import re
import json

with open("./contact/val.json") as f:
    data = json.load(f)    
res = list(data.values())

sphere2=0
spherecount2=0
cone2=0
conecount2=0
cube2=0
cubecount2=0
cylinder2=0
cylindercount2=0
torus2=0
toruscount2=0
invertedcone2=0
invertedconecount2=0
sidedcylinder2=0
sidedcylindercount2=0
for i in range(len(res)):
    numofobject=len(res[i])
#     print(res[i])
    for b in range(len(res[i])):
        obj2=res[i][b][0]
        obj3=res[i][b][2]
        obj = re.sub(r'[^a-zA-Z ]+', '', obj2)
        if obj == 'Sphere':
            sphere2 = sphere2 +1
            if str(obj3) == '[1]':
                spherecount2=spherecount2+1
                
        elif obj == 'Cone':
            cone2=cone2 +1
            if str(obj3) == '[1]':
                conecount2=conecount2+1
                
        elif obj == 'Cube':
            cube2=cube2 +1
            if str(obj3) == '[1]':
                cubecount2=cubecount2+1
                
        elif obj == 'Cylinder':
            cylinder2=cylinder2 +1
            if str(obj3) == '[1]':
                cylindercount2=cylindercount2+1
                
        elif obj == 'Torus':
            torus2=torus2 +1
            if str(obj3) == '[1]':
                toruscount2=toruscount2+1
                
        elif obj == 'InvertedCone':
            invertedcone2=invertedcone2 +1
            if str(obj3) == '[1]':
                invertedconecount2=invertedconecount2+1
                
        elif obj == 'SideCylinder':
            sidedcylinder2=sidedcylinder2 +1
            if str(obj3) == '[1]':
                sidedcylindercount2=sidedcylindercount2+1
                
print("Sphere:"+ str(sphere2))
print("Sphere count:"+ str(spherecount2))

print("Cone:"+ str(cone2))
print("Cone count:"+ str(conecount2))

print("Cube:"+ str(cube2))
print("Cube count:"+ str(cubecount2))

print("Cylinder:"+ str(cylinder2))
print("Cylinder count:"+ str(cylindercount2))

print("Torus:"+ str(torus2))
print("Torus count:"+ str(toruscount2))

print("InvertedCone:"+ str(invertedcone2))
print("InvertedCone count:"+ str(invertedconecount2))

print("SideCylinder:"+ str(sidedcylinder2))
print("SideCylinder count:"+ str(sidedcylindercount2))


# In[4]:


import csv
import re
import json

with open("./contact/test.json") as f:
    data = json.load(f)    
res = list(data.values())

sphere3=0
spherecount3=0
cone3=0
conecount3=0
cube3=0
cubecount3=0
cylinder3=0
cylindercount3=0
torus3=0
toruscount3=0
invertedcone3=0
invertedconecount3=0
sidedcylinder3=0
sidedcylindercount3=0
for i in range(len(res)):
    numofobject=len(res[i])
#     print(res[i])
    for b in range(len(res[i])):
        obj2=res[i][b][0]
        obj3=res[i][b][2]
        obj = re.sub(r'[^a-zA-Z ]+', '', obj2)
        if obj == 'Sphere':
            sphere3 = sphere3 +1
            if str(obj3) == '[1]':
                spherecount3=spherecount3+1
                
        elif obj == 'Cone':
            cone3=cone3 +1
            if str(obj3) == '[1]':
                conecount3=conecount3+1
                
        elif obj == 'Cube':
            cube3=cube3 +1
            if str(obj3) == '[1]':
                cubecount3=cubecount3+1
                
        elif obj == 'Cylinder':
            cylinder3=cylinder3 +1
            if str(obj3) == '[1]':
                cylindercount3=cylindercount3+1
                
        elif obj == 'Torus':
            torus3=torus3 +1
            if str(obj3) == '[1]':
                toruscount3=toruscount3+1
                
        elif obj == 'InvertedCone':
            invertedcone3=invertedcone3 +1
            if str(obj3) == '[1]':
                invertedconecount3=invertedconecount3+1
                
        elif obj == 'SideCylinder':
            sidedcylinder3=sidedcylinder3 +1
            if str(obj3) == '[1]':
                sidedcylindercount3=sidedcylindercount3+1
                
print("Sphere:"+ str(sphere3))
print("Sphere count:"+ str(spherecount3))

print("Cone:"+ str(cone3))
print("Cone count:"+ str(conecount3))

print("Cube:"+ str(cube3))
print("Cube count:"+ str(cubecount3))

print("Cylinder:"+ str(cylinder3))
print("Cylinder count:"+ str(cylindercount3))

print("Torus:"+ str(torus3))
print("Torus count:"+ str(toruscount3))

print("InvertedCone:"+ str(invertedcone3))
print("InvertedCone count:"+ str(invertedconecount3))

print("SideCylinder:"+ str(sidedcylinder3))
print("SideCylinder count:"+ str(sidedcylindercount3))


# In[5]:


totalsphere=sphere+sphere2+sphere3
totalsphereacc=spherecount+spherecount2+spherecount3
print(totalsphere)
totalcone=cone+cone2+cone3
totalconeacc=conecount+conecount2+conecount3
print(totalcone)
totalcube=cube+cube2+cube3
totalcubeacc=cubecount+cubecount2+cubecount3
print(totalcube)
totalcyl=cylinder+cylinder2+cylinder3
totalcylinderacc=cylindercount+cylindercount2+cylindercount3
print(totalcyl)
totaltor=torus+torus2+torus3
totaltoracc=toruscount+toruscount2+toruscount3
print(totaltor)
totalinvertedcone=invertedcone+invertedcone2+invertedcone3
totalinvertedconeacc=invertedconecount+invertedconecount2+invertedconecount3
print(totalinvertedcone)
totalsidedcylinder=sidedcylinder+sidedcylinder2+sidedcylinder3
totalsidedcylinderacc=sidedcylindercount+sidedcylindercount2+sidedcylindercount3
print(totalsidedcylinder)

print("Sphere acc:"+str(totalsphereacc/totalsphere))
print("Cone acc:"+str(totalconeacc/totalcone))
print("Cube acc:"+str(totalcubeacc/totalcube))
print("Cylinder acc:"+str(totalcylinderacc/totalcyl))
print("Torus acc:"+str(totaltoracc/totaltor))
print("Inverted Cone acc:"+str(totalinvertedconeacc/totalinvertedcone))
print("Sided Cylinder acc:"+str(totalsidedcylinderacc/totalsidedcylinder))


# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = {'Sphere':totalsphere, 'Cone':totalcone, 'Cube':totalcube,
        'Cylinder':totalcyl,'Torus':totaltor,'InvertedCone':totalinvertedcone, 'SidedCylinder':totalsidedcylinder}
courses = list(data.keys())
print(courses)
values = list(data.values())
acclist=[totalsphereacc/totalsphere,totalconeacc/totalcone,totalcubeacc/totalcube,totalcylinderacc/totalcyl,totaltoracc/totaltor,totalinvertedconeacc/totalinvertedcone,totalsidedcylinderacc/totalsidedcylinder]
accstable=[0.08740831295843521,0.44922879177377895,0.47734326505276226,0.40717029449423814,0.37639311043566365,0,0.09225092250922509]
acccontain=[0.6400966183574879,0.5206662553979026,0.4742671009771987,0.5175699821322216,0.711324570273003,0.7560823456019963,0.5876747141041931]
print(acclist)

IT = [1656, 1621, 1535, 1679, 1978,1603,1574]
ECE = [1636, 1556, 1611, 1562, 1974,1623,1626]
CSE = [1644, 1612, 1596, 1636, 1948,1567,1646]
barWidth = 0.25
# Set position of bar on X axis
br1 = np.arange(len(IT))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]


# create figure and axis objects with subplots()
# create figure and axis objects with subplots()

fig,ax = plt.subplots(figsize =(12, 8))


# make a plot
# ax.bar(courses, values, color ='maroon',
#         width = 0.4)

# ax.bar(range(len(courses)), values,color ='maroon', align='edge', width=0.4 )

ax.bar(br1, IT, color ='c', width = barWidth,
        edgecolor ='grey', label ='Containment')
ax.bar(br2, ECE, color ='limegreen', width = barWidth,
        edgecolor ='grey', label ='Stability')
ax.bar(br3, CSE, color ='y', width = barWidth,
        edgecolor ='grey', label ='Contact')

acccontain
# set x-axis label
ax.set_xlabel("Classes of objects",fontsize=16)
ax.tick_params(axis='x', which='major', labelsize=11)


# set y-axis label
ax.set_ylabel("Number of objects",color="blue",fontsize=16)
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(courses, acccontain,"crimson",linestyle='dotted',label='Containment',linewidth=3,marker="o", markersize=7)

ax2.plot(courses, accstable,color='crimson',linestyle='dashed',label='Stability',linewidth=3,marker="o", markersize=7)
ax2.plot(courses, acclist,'crimson',label='Contact',linewidth=3,marker="o", markersize=7 )
ax2.set_ylabel("Success rate of the physical interactions",color="crimson",fontsize=16)
plt.ylim([0, 1])
# plt.title("Data Analysis of SPACE dataset",fontsize=20)

ax.legend(loc=1)
ax2.legend(loc=4)
plt.show()
# save the plot as a file
fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg',
            format='jpeg',
            dpi=100,
            bbox_inches='tight')

