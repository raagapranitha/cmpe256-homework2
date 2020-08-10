import json
import os
f = open(os.getcwd()+'/bundle_data.json','r')
data = f.read()
f.close()
f2 = open(os.getcwd()+'/temp.txt','w')
f2.write((data))
f2.close()

f = open(os.getcwd()+'/temp.txt','r')
lines = f.readlines()
f.close()
f1 = open(os.getcwd()+'/temp1.txt','w')
f1.write('[')
for line in lines:
	temp = line+','
	f1.writelines((temp))
f1.write(']')
f1.close() 	