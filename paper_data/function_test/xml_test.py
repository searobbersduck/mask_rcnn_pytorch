import xml.etree.ElementTree as ET

src = '/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/data/1233_left_512.xml'

tree = ET.parse(src)

root = tree.getroot()

for obj in tree.getiterator('object'):
    for child in obj:
        print(child)

print(root)

