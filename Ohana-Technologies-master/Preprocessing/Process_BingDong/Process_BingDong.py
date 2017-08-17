import csv

with open('sample.csv') as csv_file:
    spam_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
    for row in spam_reader:
        print (row)