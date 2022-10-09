import os
import csv
from anno_document import *

def statistics(root_path, file):
    statistics_dict = {}
    labelfiles = os.listdir(root_path)
    for labelfile in labelfiles:
        print(labelfile)
        label_path = root_path + labelfile
        with open(label_path, 'r') as f:
            reader = csv.reader(f)
            for i in reader:
                class_ = list(file.keys())[list(file.values()).index(int(float(i[1])))]
                if class_ in statistics_dict.keys():
                    statistics_dict[class_] = statistics_dict[class_] + 1
                else:
                    statistics_dict[class_] = 1

    print(statistics_dict)

if __name__ == '__main__':
    root_path = ''
    statistics(root_path, file_mabing)