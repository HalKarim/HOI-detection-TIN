
import numpy as np
a=[]
b=[]
for c in range(40):
    if c<20:
        a.append(c)
d=np.array(a)
print (a)
print (d)
for i in range(0, len(a),1):
    b.append(a[i:i+1])
e=np.array(b)
print (b)

