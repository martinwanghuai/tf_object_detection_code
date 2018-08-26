'''
Created on 26 Aug 2018

@author: martinwang
'''
import os
import xml.dom.minidom
import pandas as pd

 
# ANNOTATIONS_DIR_PREFIX = "/Users/martinwang/eclipse-workspace/tf_object_detection_code/annotations/xmls/"
ANNOTATIONS_DIR_PREFIX = "/Users/martinwang/Desktop/temp/annotations/xmls/"
SAVE_FILE = "/Users/martinwang/Desktop/temp/DataAnalyze_Tank.txt";
def write_result(str):
    with open(SAVE_FILE,'a') as file:
        file.write(str+'\n')
        file.close()


def read_file(file_path):
    file_prefix = file_path.split(".xml")[0]
    
    dom = xml.dom.minidom.parse(os.path.join(ANNOTATIONS_DIR_PREFIX,file_path))
    root = dom.documentElement

    instances_temp = root.getElementsByTagName('name')
    instances = [elem.firstChild.data for elem in instances_temp]
    classes = pd.Series(instances).unique()
    
    write_result(file_prefix + "\t" + str(classes.size) + "\t" + str(len(instances)))

write_result("File\tClasses\tObject\t")
for filename in os.listdir(ANNOTATIONS_DIR_PREFIX):
    if filename.endswith('xml'):
        read_file(filename)