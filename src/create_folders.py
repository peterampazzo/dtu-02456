# go through the data split file and put the corresponding annotation from part_1 in a correct folder (train, test, validation)
import csv
import shutil
import os

training, test, val = [], [], []

filename = 'data_split.csv'

with open(filename, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    next(datareader)
    for annot, split_type in datareader:
        
        # get the annotation file
#         print(annot)
        # get the type of folder it should be in
#         print(split_type)
    
        # fetch the annotation from part_1 folder and put it in the right folder 
#         for foldername in os.listdir('part_1'):
        folder_source = f'part_1/{annot}/'
#             if foldername == annot:
                #put the annotation in the right folder
        if (split_type == 'training'):
            training.append(annot)
            folder_destination = 'Myanmar_data/training_set/image/'
            shutil.move(folder_source, folder_destination)

        elif (split_type == 'test'):
            test.append(annot)
            folder_destination = 'Myanmar_data/test_set/image/'
            shutil.move(folder_source, folder_destination)


        elif (split_type == 'validation'):
            val.append(annot)
            folder_destination = 'Myanmar_data/val_set/image/'
            shutil.move(folder_source, folder_destination)
#             else:
#                 continue


print(training)
print(test)
print(val)