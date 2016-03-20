#from subprocess import call
#call("java -cp moa.jar -javaagent:sizeofag.jar moa.DoTask \"EvaluatePrequential -l trees.HoeffdingTree -s (ArffFileStream -f C:/Users/HP-PC/Desktop/final-year-project/data/sear.data)\" -o op.csv")
import pandas as pd

df = pd.read_csv("C:/Users/HP-PC/Desktop/final-year-project/op.csv")
count=0
for index,row in df.iterrows():
    if row[0] == row[1]:
        count = count + 1
print(count/600)
