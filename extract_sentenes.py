import os

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk import word_tokenize
from annot_util.brat_util import Document, ent_cmp, Corpus, ENCODING
from collections import deque
from annot_util.config import ChemProtConfig
import itertools
import random

from collections import defaultdict

import warnings
import codecs


punkt_tokenizer = PunktSentenceTokenizer()
punkt_tokenizer._params.abbrev_types.add('dr')  # ref: http://nlpforhackers.io/splitting-text-into-sentences/

config = ChemProtConfig('config/main_config.ini')

cpr_labels = config.get_target_labels()

import re
par_ptn = re.compile(r'.*\((.*)\)')

print "Candidate cpr types:"
print cpr_labels

TK_DIST_THRES = 40

def find_entity_token(tokens, entity_text):

    to_find = entity_text.split()[-1]

    if '(' in entity_text or ')' in entity_text:
        # e.g. p27(kip1) -> p27 ( kip1 ) -> kip1
        res = par_ptn.search(entity_text)
        if res is None:
            print "error pattern: \"" + entity_text + "\" tokens: " + str(tokens)
        else:
            to_find = res.group(1)

    for i, tk in enumerate(tokens):

        if tk == to_find:
            return i

        if '-' not in tk:
            if tk == to_find:
                return i
        else:   # contains '-', e.g. "[cyclin E]-[cdk2]"
            if to_find == tk.split('-')[0]:
                # print "find \"%s\" from \"%s\"\t%d" % (to_find, tk, max(i - 1, 0))
                return max(i - 1, 0)
            elif to_find == tk.split('-')[1]:
                # print "find \"%s\" from \"%s\"\t%d" % (to_find, tk, i + 1)
                return i + 1

    return -1


def find_entity_token_from_tag_sent(sent, ent):
    ent_token = ent.get_text().split()[-1]
    for i, tk in enumerate(sent.sent_tag_tokens):
        if 'ENTITY_' + ent.get_id() in tk:
            return i

        if ent_token == tk:
            return i

        if tk in ent_token:
            return i

    print "error: token not found from tags: %s, %s \"%s\"" % (sent.doc_id, ent.get_id(), ent.get_text())
    print sent.sent_tag_tokens

    return -1


class Sentence:
    id = 0
    sent_ent_type_text = ""
    sent_tag_text = ""
    sent_tag_tokens = []
    sent_ent_type_tokens = []

    # TODO: Merge to Annotation
    def __init__(self, sent_span, sent_text, sent_offset=0, doc_id='na'):
        """
            sent_span = list(punkt_tokenizer.span_tokenize(text))
            sent_text = list(punkt_tokenizer.sentences_from_text(text))

        :param sent_span: (begin, end)
        :param sent_text: str
        :param sent_offset:
        """
        self.doc_id = doc_id
        self.sent_offset = sent_offset
        self.sent_span = (sent_offset + sent_span[0], sent_offset + sent_span[1])
        self.sent_text = sent_text
        self.tokens = word_tokenize(sent_text)

    def get_begin(self):
        return self.sent_span[0]

    def get_end(self):
        return self.sent_span[1]

    def get_text(self):
        return self.sent_text

    def __repr__(self):
        return "%d,%d\t%s" % (self.sent_span[0], self.sent_span[1], self.sent_text)


class CPRDoc():
    entity_list = []

    def __init__(self, doc, verbose=False):

        # TODO: use inheritance
        self.doc_id = doc.doc_id
        self.text = doc.text
        self.annot = doc.annot
        self.entities = doc.entities
        self.relations = doc.relations     # key: source entity, value: list of target entity and relations
        self.relations_dict = doc.relations_dict

        # print self.relations_dict

        self.verbose = verbose
        self.sents = []
        self.detect_sentences()

        self.entity_list = self.entities.values()
        # need to sort entities from dict
        self.entity_list.sort(cmp=ent_cmp)
        # print self.doc_id
        # print self.entity_list
        # print len(self.entity_list)
        # for ent in self.entities:
        #     print ent
        # print self.entity_list

        self.sent_entities = []
        self.align_sent_entities()
        self.replace_sent_entities()

    def detect_sentences(self):
        """
        Run sentence detector
        Save sentences and IDs in CPRDoc
        :return:
        """
        line_offset = 0
        for line in self.text.split('\n'):
            sent_spans = list(punkt_tokenizer.span_tokenize(line))
            sent_texts = list(punkt_tokenizer.sentences_from_text(line))

            self.sents += [Sentence(sent_span, sent_text, line_offset, doc_id=self.doc_id)
                           for sent_span, sent_text in zip(sent_spans, sent_texts)]

            line_offset += len(line) + 1    # +1 to compensate split character ('\n')

        sent_id = 0
        for sent in self.sents:
            sent.id = sent_id
            sent_id += 1

    def get_relation(self, src_ent_id, target_ent_id):
        """
        Look up relation by src id and target id
        :param src_ent_id:
        :param target_ent_id:
        :return:
        """
        rel = self.relations_dict[src_ent_id].get(target_ent_id)
        return rel

    def detect_sent_relation_candidates(self, fo=None):
        """
        Find within-sentence relations
        :param fo:
        :param print_format:
        :return:
        """
        # loop among sentents
        for (_, cur_sent, ent_list) in self.sent_entities:

            if len(ent_list) == 0:
                continue

            # loop among all candidate pairs
            for src, target in itertools.permutations(ent_list, 2):

                src_id = src.get_id()
                target_id = target.get_id()

                # decide if the two entites have different types
                # src.type ={"CHEMICAL", "GENE-N", "GENE-Y"}
                if src.type[0] == target.type[0]:
                    continue

                # skip GENE-Y and GENE-N entities as src
                if src.type != 'CHEMICAL':
                    continue

                target_rel = self.get_relation(src_id, target_id)

                # tokens = self.sents[src.sent_id].tokens

                tokens = self.sents[src.sent_id].sent_ent_type_tokens
                # Write none relations
                if target_rel is None:

                    out_list = self.print_na_relation(src, target, self.sents[src.sent_id])

                    if out_list is None:
                        continue

                    if fo is not None:
                        fo.write('\t'.join(out_list) + '\n')
                    else:
                        print '\t'.join(map(unicode.lower, out_list))

                    continue

                rel_type = target_rel.get_type()

                # src_tk_id = find_entity_token(tokens, src.get_text())
                # target_tk_id = find_entity_token(tokens, target.get_text())

                src_tk_id = find_entity_token_from_tag_sent(cur_sent, src)
                target_tk_id = find_entity_token_from_tag_sent(cur_sent, target)

                # if abs(src_tk_id - target_tk_id) > TK_DIST_THRES:
                #     return

                out_list = [rel_type,
                            src.get_type(), target.get_type(),
                            src_tk_id,
                            target_tk_id,
                            ' '.join(tokens),
                            src_id, target_id,
                            src.doc_id
                            ]

                if src_tk_id == -1 or target_tk_id == -1:
                    # print problematic relations
                    print "Uncaptured instance in %s: (%s: %s -> %s: %s)" % \
                          (self.doc_id, src_id, src.get_text(), target_id, target.get_text())
                    print "\t" + str(out_list)
                    print "\t" + str(tokens)
                    continue

                out_list = map(unicode, out_list)

                if fo is not None:
                    fo.write('\t'.join(out_list) + '\n')
                else:
                    print '\t'.join(map(unicode.lower, out_list))

    def replace_sent_entities(self):
        """
        Find entities in the sentence and replace with tags.
        :param fo:
        :param print_format:
        :return:
        """
        # loop among sentences
        for (_, cur_sent, ent_list) in self.sent_entities:

            if len(ent_list) == 0:
                continue

            sent_tag_text = cur_sent.sent_text
            sent_ent_type_text = cur_sent.sent_text
            for ent in ent_list:
                type_str = ent.get_type().lower() if ent.get_type() == "CHEMICAL" else "gene"
                try:
                    sent_tag_text = re.sub(re.escape(ent.get_text()), 'ENTITY_%s' % ent.get_id(), sent_tag_text, count=1)
                    sent_ent_type_text = re.sub(re.escape(ent.get_text()), type_str, sent_ent_type_text, count=1)

                except:
                    print "error occurs when replacing \"%s\" from \"%s\"" % (ent.get_text(), sent_tag_text)

            # print sent_tag_text
            cur_sent.sent_tag_text = sent_tag_text
            # sent_tag_text = re.sub('[-//]', ' ', sent_tag_text)
            # sent_ent_type_text = re.sub('[-//]', ' ', sent_ent_type_text)

            cur_sent.sent_tag_tokens = word_tokenize(sent_tag_text)
            cur_sent.sent_ent_type_tokens = word_tokenize(sent_ent_type_text)
            # print cur_sent.sent_tag_tokens


    def print_na_relation(self, src, target, cur_sent):
        tokens = cur_sent.sent_ent_type_tokens
        # target_rel = self.get_relation(src.get_id(), target.get_id())
        # rel_id = "na@r@%s@gold" % self.doc_name
        rel_type = 'NA'
        # src_tk_id = find_entity_token(tokens, src.get_text())
        # target_tk_id = find_entity_token(tokens, target.get_text())

        src_tk_id = find_entity_token_from_tag_sent(cur_sent, src)
        target_tk_id = find_entity_token_from_tag_sent(cur_sent, target)

        if src_tk_id == -1 or target_tk_id == -1:
            return

        if abs(src_tk_id - target_tk_id) > TK_DIST_THRES:
            return

        out_list = [rel_type,
                    src.get_type(), target.get_type(),
                    # src.get_text().split(' ')[-1], target.get_text().split(' ')[-1],
                    src_tk_id,
                    target_tk_id,
                    ' '.join(tokens),
                    src.get_id(), target.get_id(),
                    src.doc_id
                    ]

        # out_list = map(str, out_list)
        out_list = map(unicode, out_list)

        return out_list

    def align_sent_entities(self):
        """
        Align sentences and entities.
        :return:
        """

        sent_queue = deque(self.sents)

        cur_sent = sent_queue.popleft()

        cur_sent_ent_list = []

        for ent in self.entity_list:

            # move to next sentence until the entity and the sentence are aligned
            while ent.get_begin() >= cur_sent.get_end():
                # if the entity list is not empty, store it in result list
                if len(cur_sent_ent_list) > 0:
                    self.sent_entities.append((cur_sent.id, cur_sent, cur_sent_ent_list))
                    cur_sent_ent_list = []

                cur_sent = sent_queue.popleft()

            if ent.get_end() <= cur_sent.get_end():
                # entity within sentence, store, move to next entity
                ent.sent_id = cur_sent.id
                cur_sent_ent_list.append(ent)
                continue


def do_corpus(corpus_root, out_path, is_testing=False):

    corpus = Corpus(corpus_root, is_test=is_testing)

    with codecs.open(out_path, 'w', encoding=ENCODING) as fo:
        for doc in corpus.docs.itervalues():
            cpr_doc = CPRDoc(doc)
            cpr_doc.detect_sent_relation_candidates(fo)

        print "Written to: " + out_path

def do_cpr():

    in_root = config.get('main', 'corpus_dir')
    out_root = config.get('main', 'out_dir')

    do_corpus(os.path.join(in_root, 'chemprot_training'), os.path.join(out_root, 'training.txt'))
    do_corpus(os.path.join(in_root, 'chemprot_development'), os.path.join(out_root, 'development.txt'))
    do_corpus(os.path.join(in_root, 'chemprot_test'), os.path.join(out_root, 'test.txt'), is_testing=False)


if __name__ == '__main__':
    do_cpr()
    # debug_cpr()
