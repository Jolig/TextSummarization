"""
Creates "data.csv" by taking all the documents in the mentioned directory using os.walk
Author - Devyani
"""

import os

from helper import dissect

# Directory where all the data files are present
data = "/Users/akhila/Desktop/IIITB/1-2/Text/AlleDocs/Data"
datafilepath_tobe_stored = "/Users/akhila/Downloads/data.csv"


# List which saves all the paragraphs of each file present in the data directory
datalist = ["Paragraph"]


# walks through the directory and sub directories, reads the file data and append to the datalist
for dirName, subdirList, fileList in os.walk(data):
    for fname in fileList:
        if (fname[0] != '.'): # ignore any hidden files
            with open(dirName+"/"+fname, 'r', encoding="utf-8") as f:
                # read the file and add to the datalist
                content = f.read()
                content = content.encode('ascii', 'ignore').decode('ascii')
                datalist.append(content)
                print(content)

print("length", len(datalist))


# exports the datalist to csv file at specified location
dissect.export(datalist, datafilepath_tobe_stored)
