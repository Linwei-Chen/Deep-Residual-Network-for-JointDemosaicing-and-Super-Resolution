def save_txt(filename,contents):
    fh = open(filename, 'w')#, encoding='utf-8')
    fh.write(contents)
    fh.close()


train_data_path="/Users/linweichen/Desktop/计算机学习资料/TrainData/RAISE_1K/Train_Data.txt"
test_data_path="/Users/linweichen/Desktop/计算机学习资料/TrainData/RAISE_1K/Test_Data.txt"

data_dir = "/Users/linweichen/Desktop/计算机学习资料/TrainData/RAISE_1K/Bayer/Bayer"
label_dir = "/Users/linweichen/Desktop/计算机学习资料/TrainData/RAISE_1K/Resize/real"

contents = ""
for i in range(1, 1001):
    contents += data_dir + str(i) + ".TIF" + " " + label_dir + str(i) + ".TIF\n"

print(contents)

save_txt(train_data_path,contents)

'''

contents=""

for i in range(901, 1001):
    contents += data_dir + str(i) + ".TIF" + " " + label_dir + str(i) + ".TIF\n"

print(contents)




save_txt(test_data_path,contents)


data_dir = "/Users/linweichen/Desktop/计算机学习资料/TrainData/RAISE_1K/Fast_test/"

contents = ""
for i in range(1, 1001):
    contents += data_dir + str(i) + "bayer" + ".TIF" + " " + data_dir + str(i) + "real" + ".TIF\n"

print(contents)

fast_test_data_path = "/Users/linweichen/Desktop/计算机学习资料/TrainData/RAISE_1K/Fast_Test_Data.txt"

save_txt(fast_test_data_path, contents)
'''
