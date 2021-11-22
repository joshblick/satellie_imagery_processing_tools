import tensorflow as tf
import argparse
import os, glob
import xml.etree.ElementTree as ET
import pandas as pd

# defining constants to come from the command line
parser = argparse.ArgumentParser()

parser.add_argument("--annotation_location",
                    default=os.getcwd(),
                    help="location where tfrecords and pbtext will be placed, defaults to cwd")

parser.add_argument("--imagesxml_location",
                    default=os.getcwd(),
                    help="location where csv of the labels will be placed, defaults to cwd")

parser.add_argument("--debug_csv_location",
                    default=False,
                    help="location where csv of labels will be placed, otherwise doesn't generate")
                    
def create_label_dataframe(imagesxml_location):
    """Form a pandas dataframe
    
    Keyword arguments:
    imagesxml_location --  the location of the images and xml label files
    
    
    """
    #iterate through each xml, obtain desired information
    
    xml_list = []
    for xml_file in glob.glob(os.path.join(imagesxml_location) + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
        #adding something to place a row when there are no objects
        if len(root.findall('object'))==0:
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     None,
                     None,
                     None,
                     None,
                     None
                     )
            xml_list.append(value)

    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    args = parser.parse_args()
    
    label_dataframe = create_label_dataframe(args.imagesxml_location)
    
    print("annotations will be placed in\n{}".format(args.annotation_location))
    
    if args.debug_csv_location!=False: #TODO: work out cleaner way to do this double negative
        label_dataframe.to_csv(os.path.join(args.debug_csv_location)+"\debug.csv", index=False)
if __name__=="__main__":
    main()