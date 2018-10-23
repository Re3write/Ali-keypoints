import os

rootdir='D:\workplace\常远\常远测试集'
picdir=os.listdir(rootdir)

ori=os.listdir('D:\workplace\常远\常远\测试集')
final=[]

for dir in picdir:
    tempdir_path=os.path.join(rootdir,dir)
    tempdir=os.listdir(tempdir_path)
    for pic in tempdir:
        if ori.count(pic)==1:
            final.append(pic)
        else:
            print(pic)

print(len(final))