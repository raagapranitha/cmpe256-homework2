f = open('/Users/raagapranithakolla/sjsu/cmpe256/homework2/temp.txt','r')
lines = f.readlines()
f.close()
f1 = open('/Users/raagapranithakolla/sjsu/cmpe256/homework2/video_bundle_data2.json','w')
f1.write('[')
for idx,line in enumerate(lines):
	if idx == 0:
		temp = line
	else:
		temp = line+','
	temp = temp.replace("{'",'{"')
	temp = temp.replace("':",'":')
	temp = temp.replace(": '",': "')
	temp = temp.replace("',",'",')
	temp = temp.replace(", '",', "')
	temp = temp.replace("'}",'"}')
	f1.writelines((temp))
f1.write(']')
f1.close()