import os

def get_filelist(dir):
    for dirs in os.walk(dir):
        print("#######dir list#######")   #dirs中存着所有的目录文件
        if dirs[0]==dir:   #为了避免输出和根目录一样的格式
            continue
        else:

            source = dirs[0]
            print(source)
            # for i in range(len(source)):
            #     source = dirs[0]
            #     print(source[i])
        # print(dirs[2])
        # for dir in dirs:
        #     print(dir)
        # print("#######dir list#######")
        #
        # print("#######file list#######")
        # # for filename in files:
        #     print(filename)
        #     fullname = os.path.join(home, filename)
        #     print(fullname)
        # print("#######file list#######")

if __name__ == "__main__":
    get_filelist('/home/atr2/data/car/2')