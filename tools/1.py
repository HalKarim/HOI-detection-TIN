import pickle

#dic=pickle.load(open("80000_TIN_D_noS.pkl","rb"),encoding="latin1")
dic={1,2,3}
dic=pickle.decode(dic,"latin1")
pickle.dump(dic,open("1.pkl","wb"),protocol=2)
#open("/home/zhw/1.txt","wb")

import pickle
fp = open("6000_TIN_VCOCO_0.6_0.4_naked.pkl", "rb")
ngo = pickle.load(fp)
ngo=list(ngo)
print("writing to 1.pkl")
pickle.dump(ngo, open("1.pkl", "wb"),protocol=2)
