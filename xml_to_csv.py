# python xml_to_csv.py

import glob
import pandas
import xml.etree.ElementTree as ET
import os


def xml_to_csv(path):
    print('path: ' + path)
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        print('xml_file: ' + xml_file)
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
                     int(member[4][3].text))
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    return pandas.DataFrame(xml_list, columns=column_name)


def main():
    for directory in ['train', 'test']:
        xml_path = os.path.join(os.getcwd(), 'xml/{}'.format(directory))
        xml_df = xml_to_csv(xml_path)
        xml_df.to_csv('data/{}_labels.csv'.format(directory), index=None)
        print('Successfully converted xml to csv.')
    # print("Total " + str(i) + " xml files has been converted.")


if __name__ == '__main__':
    main()
