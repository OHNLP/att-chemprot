import os
import codecs


def readAnn(textfolder = "data/dev/"):
    '''
    Read .ann files and look up corresponding spans in .txt files
    :param textfolder:
    :return:
    '''

    flist = os.listdir(textfolder)
    for f in flist:
        f_anno = codecs.open(os.path.join(textfolder, f), "rU")
        f_text = codecs.open(os.path.join(textfolder, f.replace(".ann", ".txt")), "rU")

        # there's only one line, as each .ann file is one text paragraph
        for l in f_text:
            text = l

        for l in f_anno:
            anno_inst = l.strip().split("\t")
            if len(anno_inst) == 3:
                keytype, start, end = anno_inst[1].split(" ")
                if not keytype.endswith("-of"):

                    # look up span in text and print error message if it doesn't match the .ann span text
                    keyphr_text_lookup = text[int(start):int(end)]
                    keyphr_ann = anno_inst[2]
                    if keyphr_text_lookup != keyphr_ann:
                        print("Spans don't match for anno " + l.strip() + " in file " + f)


def prep_word(word):
    word = word.lower()
    if word.endswith('s'):
        word = word[:-1]

    return word

class AnnotItem:
    annot_type = "T"

    start = 0
    end = 0
    label = 'Process'

    def __init__(self, line):

        try:
            self.line = line
            self.anno_inst = line.strip().split("\t")
            self.annot_id = self.anno_inst[0]
            if not self.annot_id.startswith("T"):
                self.annot_type = 'R'
                self.label = self.anno_inst
            else:
                self.label, start, end = self.anno_inst[1].split(" ")
                self.start = int(start)
                self.end = int(end)
                self.mention = self.anno_inst[2]
        except ValueError as err:
            print err.message
            print line

    def __repr__(self):
        return self.line
