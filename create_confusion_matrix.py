import pandas as pd
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from matplotlib import rcParams



# path = './data/111.csv'  #CAST
# path = './data/viT.csv'    #vit
# path = './data/swinT.csv'     #原始st
path = './swin_transformer/IST-GEI-front.csv'
config = {
    "font.family":'Times New Roman',  # 设置字体类型
}

rcParams.update(config)
# 使用pandas读入
data = pd.read_csv(path) #读取文件中所有数据
print(data)
# 按列分离数据
x = data[['Actual label', 'Predict labels']]#读取某两列
print(x)
# y_true = data[['y_true']]#读取某一列
# y_pred = data[['y_pred']]
true = data['Actual label']#读取某一列
pred = data['Predict labels']
y_true = true.values.tolist()
y_pred = pred.values.tolist()
print(y_true)
print(y_pred)



classes = list(set(y_true))
classes.sort()
confusion = confusion_matrix(y_pred, y_true)
plt.imshow(confusion, cmap=plt.cm.Purples)
indices = range(len(confusion))
plt.xticks(indices, classes)
plt.yticks(indices, classes)
plt.colorbar()
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.ylabel('True label', fontsize=16,fontfamily="TimesNewRoman")
plt.xlabel('Predicted label', fontsize=16,fontfamily="TimesNewRoman")
for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):
        plt.text(first_index, second_index, confusion[first_index][second_index],fontsize = 20,fontfamily="TimesNewRoman",va = 'center', ha = 'center',color="white" if first_index==0 and second_index == 0 else "black")

plt.savefig('IST-GEI-front.png', format='png')
plt.show()

