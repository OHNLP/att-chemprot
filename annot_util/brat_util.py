"""
    File name: brat_util.py
    Project: Semi-IR
    Desciption:
    Author: Sijia Liu (m142167)
    Date created: Sept 18, 2017
"""
import os
import codecs
from warnings import warn
from collections import defaultdict


ENCODING = 'utf-8'

def ent_cmp(x, y):
    """
    Compare first span of Entity
    :param x:
    :param y:
    :return:
    """
    if not (isinstance(x, Entity) and isinstance(y, Entity)):
        print "Warning: unacceptable comaparion. Returned 0s"
        return 0

    if x.get_spans()[0][0] - y.get_spans()[0][0] is not 0:
        return x.get_spans()[0][0] - y.get_spans()[0][0]

    return x.get_spans()[0][1] - y.get_spans()[0][1]



class Entity():

    sent_id = 0

    def __init__(self, splits):
        self.doc_id = splits[0]
        self.ent_id = splits[1]
        self.type = splits[2]
        self.span = (int(splits[3]), int(splits[4]))
        self.annot_text = splits[5]

    def get_spans(self):
        return [self.span]

    def get_begin(self):
        return self.span[0]

    def get_end(self):
        return self.span[1]

    def get_id(self):
        return self.ent_id

    def get_text(self):
        return self.annot_text

    def get_type(self):
        return self.type

    def __repr__(self):
        return "%s\t%s" % (self.ent_id, self.annot_text)


class Relation():
    type = 'na'

    def __init__(self, splits):
        self.doc_id = splits[0]
        self.type = splits[1]
        self.src_id = splits[2][5:]
        self.target_id = splits[3][5:]

        assert self.src_id[0] == u"T" or self.target_id[0] == u"T", "Unparsable entity ID in %s" % splits

    def get_type(self):
        return self.type

class Corpus(object):
    doc_ids = set()

    def __init__(self, corpus_dir, is_test=False):
        self.basename = os.path.basename(corpus_dir)
        self.txt_path = os.path.join(corpus_dir, self.basename + '_abstracts.tsv')
        self.ent_path = os.path.join(corpus_dir, self.basename + '_entities.tsv')
        self.is_test = is_test
        if not is_test:
            self.rel_path = os.path.join(corpus_dir, self.basename + '_gold_standard.tsv')

        self.docs = dict()

        self.split_doc()


    def split_doc(self):
        with codecs.open(self.txt_path, encoding=ENCODING) as f_txt:
            for line in f_txt:
                (doc_id, title, text) = line.strip().split('\t')
                self.doc_ids |= set([doc_id])
                self.docs[doc_id] = Document(doc_id, title + '\t' + text)

        print "# of docs: %d" % len(self.docs)

        with codecs.open(self.ent_path, encoding=ENCODING) as f_ent:
            for line in f_ent:
                splits = line.strip().split('\t')
                assert len(splits) == 6
                ent = Entity(splits)
                # corpus -> doc -> entities -> entity
                self.docs[ent.doc_id].entities[ent.ent_id] = ent

                if self.docs[ent.doc_id].text[ent.span[0]: ent.span[1]] != ent.annot_text:
                    print "txt: \"%s\" -> annot:\"%s\"\t\t%s" % \
                          (self.docs[ent.doc_id].text[ent.span[0]: ent.span[1]],
                           ent.annot_text,
                           line)

        if not self.is_test:
            with codecs.open(self.rel_path, encoding=ENCODING) as f_rel:
                for line in f_rel:
                    splits = line.strip().split('\t')

                    if len(splits) == 4:
                        rel = Relation(splits)
                        self.docs[rel.doc_id].relations[rel.src_id].append(rel)
                        self.docs[rel.doc_id].relations_dict[rel.src_id][rel.target_id] = rel

                    else:
                        print "annot format error: " + line




class Document(object):
    def __init__(self, doc_id, text):

        self.doc_id = doc_id

        self.no_text = True
        self.text = text

        self.annot = []
        self.entities = dict()
        self.relations = defaultdict(list)     # key: source entity, value: list of target entity and relations
        self.relations_dict = defaultdict(dict)     # key: source entity, value: dict of target entity and relations


    def get_entities(self):
        return self.entities


    def append_text_annot(self):
        for child in self.root:
            if child.tag == 'annotations':
                self.append_text_attribute(child)

    def append_text_attribute(self, annot_node):
        # print annot_node.tag
        for entity in annot_node:
            annot_text = Entity(entity).get_text(self.text)
            entity.set('text', annot_text.strip().replace('\n', r'\n'))

    def write_new(self, out_path):
        """
        Write current XML tree to out_path
        :param out_path:
        :return:
        """
        self.pred_tree.write(out_path)

    def get_entity(self, ent_id):
        return self.entities.get(ent_id)

    # TODO: make it generalizable to other relations.
    def annot_to_str(self, delimiter='|', verbose=False):
        """
        Ad-hoc annotation print format for SSI. If there is an Event is
        related to Date, append Date to the Event string
        :param delimiter: delimiter for output fields, default as '|'
        :param verbose: flag of verbose mode
        :return:
        """
        lines = []

        for ent_id in self.entities:
            e = self.entities[ent_id]
            prop_str = e.get_prop_str()

            str_list = [self.doc_name, e.metadata['type'],
                        e.metadata['span'], e.get_text(self.text), prop_str]

            if len(str_list) != 5:
                warn("Invalid feature length: " + ent_id)
                continue

            ent_str = delimiter.join(str_list)

            if self.relations.get(ent_id) is not None:
                for rel in self.relations.get(ent_id):

                    target = self.entities[rel.get_target_id()]

                    temp_list = [target.metadata['type'],
                                 target.metadata['span'], target.get_text(self.text), target.get_prop_str()]

                    temp_str = delimiter.join(temp_list)

                    if verbose:
                        print ent_str + delimiter + temp_str

                    lines.append(ent_str + delimiter + temp_str)
            else:
                lines.append(ent_str)

        return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    corpus = Corpus('/Users/m142167/projects/bioc/chemprot_training')
