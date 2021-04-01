import os
import pandas as pd
import numpy as np
import cv2


training_data=list()
target_data=list()

folder_path=os.path.join(os.getcwd(),'UCMerced_LandUse','Images')
for directory in os.listdir(folder_path):
    temp_folder=os.path.join(folder_path,directory)
    list_files=os.listdir(temp_folder)
    for file_name in list_files:
        try:
            image_path=os.path.join(temp_folder,file_name)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dsize = (100, 100)
            output = cv2.resize(gray, dsize)
            training_data.append(output.flatten())
            target_data.append(directory)
            print("Preprocessing image ",file_name)
        
        except:
            continue
    
    

training_data=pd.DataFrame(training_data)
target_data=pd.DataFrame(target_data)
training_data.to_csv(os.path.join("data","training_data.csv"))
target_data.to_csv(os.path.join("data","target_data.csv"))
