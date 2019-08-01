import json

paths=[]
#for num in range(1,301):
#    paths.append('/home/waiyang/crowd_counting/Dataset/ShanghaiTech/part_A/train_data/images/IMG_'+str(num)+'.jpg')
#print(paths)
#with open('partA_train.json','w') as outfile:
#    json.dump(paths, outfile)
with open('part_B_val.json') as infile:
    data=json.load(infile)
for item in data:
    item=item.replace('/home/leeyh/Downloads/Shanghai/part_B_final/','/home/waiyang/crowd_counting/Dataset/ShanghaiTech/part_B/')
    paths.append(item)
with open('partB_val.json','w') as outfile:
    json.dump(paths,outfile)