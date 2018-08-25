# Script to convert yolo annotations to voc format

# Sample format
# <annotation>
#     <folder>less_selected</folder>
#     <filename>videoplayback0051.jpg</filename>
#     <size>
#         <width>1000</width>
#         <height>563</height>
#     </size>
#     <segmented>0</segmented>
#     <object>
#         <name>Tie Fighter</name>
#         <pose>Unspecified</pose>
#         <truncated>0</truncated>
#         <difficult>0</difficult>
#         <bndbox>
#             <xmin>157</xmin>
#             <ymin>165</ymin>
#             <xmax>166</xmax>
#             <ymax>176</ymax>
#         </bndbox>
#     </object>
#     <object>
#         <name>Tie Fighter</name>
#         <pose>Unspecified</pose>
#         <truncated>0</truncated>
#         <difficult>0</difficult>
#         <bndbox>
#             <xmin>136</xmin>
#             <ymin>151</ymin>
#             <xmax>145</xmax>
#             <ymax>160</ymax>
#         </bndbox>
#     </object>
# </annotation>
import os
import xml.etree.cElementTree as ET
from PIL import Image

ANNOTATIONS_DIR_PREFIX = "/Users/martinwang/Desktop/temp/label_640/"

DESTINATION_DIR = "/Users/martinwang/Desktop/temp/voc_labels"

CLASS_MAPPING = {
    '80': 'tank'
}


def create_root(file_prefix, width, height):
    root = ET.Element("annotations")
    ET.SubElement(root, "folder").text = "less_selected"
    ET.SubElement(root, "filename").text = "{}.jpg".format(file_prefix)
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(root, "segmented").text = str(0)
    return root


def create_object_annotation(root, voc_labels):
    for voc_label in voc_labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = voc_label[0]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(voc_label[1])
        ET.SubElement(bbox, "ymin").text = str(voc_label[2])
        ET.SubElement(bbox, "xmax").text = str(voc_label[3])
        ET.SubElement(bbox, "ymax").text = str(voc_label[4])
    return root


def create_file(file_prefix, width, height, voc_labels):
    root = create_root(file_prefix, width, height)
    root = create_object_annotation(root, voc_labels)
    tree = ET.ElementTree(root)
    tree.write("{}/{}.xml".format(DESTINATION_DIR, file_prefix))


def read_file(file_path):
    file_prefix = file_path.split(".txt")[0]
    image_file_name = "{}.jpg".format(file_prefix)
    img = Image.open("{}/{}".format("/Users/martinwang/Desktop/temp/image_640/", image_file_name))
    w, h = img.size
    with open(os.path.join(ANNOTATIONS_DIR_PREFIX,file_path), 'r') as file:
        lines = file.readlines()
        voc_labels = []
        for line in lines:
            voc = []
            line = line.strip()
            data = line.split()
            voc.append(CLASS_MAPPING.get(data[0]))
            center_x = float(data[1]) * w
            center_y = float(data[2]) * h
            bbox_width = float(data[3]) * w
            bbox_height = float(data[4]) * h
            voc.append(int(center_x - (bbox_width / 2)))
            voc.append(int(center_y - (bbox_height / 2)))
            voc.append(int(center_x + (bbox_width / 2)))
            voc.append(int(center_y + (bbox_height / 2)))
            voc_labels.append(voc)
        create_file(file_prefix, w, h, voc_labels)
    print("Processing complete for file: {}".format(file_path))
    
    with open("/Users/martinwang/Desktop/temp/trainval.txt",'a') as file:
        file.write(file_prefix + os.linesep)


def start():
    if not os.path.exists(DESTINATION_DIR):
        os.makedirs(DESTINATION_DIR)
    for filename in os.listdir(ANNOTATIONS_DIR_PREFIX):
                
        print("read file: {}".format(filename))
        
        if filename.endswith('txt'):
            read_file(filename)
        else:
            print("Skipping file: {}".format(filename))


if __name__ == "__main__":
    start()
