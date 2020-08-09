import json
import os
# f = open('/Users/raagapranithakolla/Downloads/bundle_data.json','r')
# data = f.read()
# f.close()
# f2 = open('/Users/raagapranithakolla/sjsu/cmpe256/homework2/temp.txt','w')
# f2.write((data))
# f2.close()

f = open('/Users/raagapranithakolla/sjsu/cmpe256/homework2/temp.txt','r')
lines = f.readlines()
f.close()
f1 = open('/Users/raagapranithakolla/sjsu/cmpe256/homework2/temp1.txt','w')
f1.write('[')
for line in lines:
	temp = line+','
	# temp = temp.replace("'b",'"b')
	f1.writelines((temp))
f1.write(']')
f1.close()
# import json
# import ast

# fr=open("/Users/raagapranithakolla/Downloads/bundle_data.json")
# fw=open("/Users/raagapranithakolla/sjsu/cmpe256/homework2/video_bundle_data.json", "w")

# for line in fr:
#     json_dat = json.dumps(ast.literal_eval(str(line)))
#     dict_dat = json.loads(json_dat)
#     json.dump(dict_dat, fw)
#     fw.write("\n")

# fw.close()
# fr.close()
# 	