'''
relationships.py: extract a relationship graph from text
'''
# pylint: disable=c-extension-no-member, import-error, wrong-import-position, invalid-name, too-many-branches, unused-import

import sys

from pathlib import Path

# Add persyn root to sys.path
sys.path.insert(0, str((Path(__file__) / '../../').resolve()))

# Color logging
from color_logging import ColorLog

log = ColorLog()

import spacy

from spacy.tokens import Doc

# import networkx as nx

# !pip install coreferee
# !python3 -m coreferee install en
# !python3 -m spacy download en_core_web_lg

import coreferee

# Merged pipeline
nlp = spacy.load('en_core_web_lg')
nlp.add_pipe('coreferee')
nlp.add_pipe('sentencizer')

# patterns = [[{"LOWER": "hackerfriendly"}]]
# attrs = {"TAG": "NNP", "POS": "PROPN", "DEP": "nsubj"}

# ruler = nlp.get_pipe("attribute_ruler")
# ruler.add(patterns=patterns, attrs=attrs)

nlp_merged = spacy.load('en_core_web_lg')
nlp_merged.add_pipe('merge_entities')
nlp_merged.add_pipe('merge_noun_chunks')

# ruler = nlp_merged.get_pipe("attribute_ruler")
# ruler.add(patterns=patterns, attrs=attrs)

archetypes = [
    "Alice", "Bob", "Carol", "Dave", "Eve",
    "Frank", "Gavin", "Heidi", "Ivan", "Judy",
    "Kaitlin", "Larry", "Mia",
    "Natalie", "Oliver", "Peggy", "Quentin", "Rupert",
    "Sophia", "Trent", "Ursula", "Victor", "Wanda",
    "Xavier", "Yolanda", "Zahara"
]

def referee(doc):
    '''
    Resolve coreferences in doc
    Returns a new doc with coreferences resolved
    '''
    if not isinstance(doc, spacy.tokens.doc.Doc):
        doc = nlp(doc)

    sent = []
    for tok in doc:
        if doc._.coref_chains is None:
            sent.append(tok.text)
            continue
        ccr = doc._.coref_chains.resolve(tok)
        if ccr is None:
            sent.append(tok.text)
        else:
            for word in ccr:
                sent.append(word.text)

    return nlp(Doc(vocab=doc.vocab, words=sent))

def find_all_modifiers(tok):
    '''
    Find all adjectives that are children of tok
    Returns a list of adjectives (text only)
    '''
    ret = []
    for child in tok.children:
        if child.pos_ == 'ADJ':
            ret.append(child.text)

    return ret

def find_all_conj(tok):
    ''' If tok is a conjunct, return all children that are appositional modifiers '''
    ret = []
    for child in tok.children:
        if child.dep_ == 'conj':
            ret = [c.text for c in child.children if c.dep_ == 'appos']
            if not ret:
                ret = [child.text] + find_all_conj(child)
    return ret

def find_all_pobj(tok):
    ''' If tok is an object of a preposition, return all children that are appositional modifiers '''
    ret = []
    for child in tok.children:
        if child.dep_ == 'pobj':
            ret = [c.text for c in child.children if c.dep_ == 'appos']
            if not ret:
                ret = [child.text] + find_all_conj(child)
    return ret

def find_all_singletons(tok):
    ''' Return a list of all descendants with only one child. '''
    if not list(tok.children):
        return []

    def all_singletons(node):
        if len(list(node.children)) > 1:
            return False

        for child in node.children:
            if not all_singletons(child):
                return False

        return True

    if not all_singletons(tok):
        return []


    ret = []
    for child in tok.children:
        ret = [child.text] + find_all_singletons(child)

    return ret

def get_relationships(doc, depth=0):
    '''
    Find all relationships in doc
    Returns a list of { left: [], rel: "", right: [] } dicts
    '''
    if depth > 1:
        print("Maximum recursion depth reached.")
        return []

    clauses = []
    ret = {
        'left': [],
        'rel': None,
        'right': []
    }

    # Merge nouns
    doc = nlp_merged(str(doc))

    # Resolve coreferences
    doc = referee(doc)

    # if doc[-1].dep_ == 'punct':
    #     ret['punct'] = doc[-1].text

    for tok in doc:

        # Find the ROOT
        if tok.dep_ != 'ROOT':
            continue

        if tok.pos_ not in ['VERB', 'AUX']:
            print("Root is not a verb, can't continue:", tok, doc)
            return []

        ret['rel'] = tok.lemma_.lower()

        if not tok.children:
            if all(ret.values()):
                return [ret]
            return []

        for child in tok.children:
            # Include modifiers (if any)
            if child.dep_ == 'neg':
                ret['rel'] = f"not {ret['rel']}"
            if child.dep_ == 'advmod':
                ret['rel'] = f"{ret['rel']} {child.lemma_.lower()}"

        for child in tok.children:
            if child.dep_ == 'nsubj':
                subj = [child.text] + find_all_conj(child)
                ret['left'] = sorted(list(set(subj)))

            elif child.dep_ == 'dobj':
                ret['right'] = [' '.join([child.text] + find_all_singletons(child) + find_all_modifiers(child))]

        # no dobj available, try something else
        if not ret['right']:
            for child in tok.children:
                # Try others
                if child.dep_ == 'acomp':
                    ret['right'] = [' '.join([child.text] + find_all_singletons(child) + find_all_modifiers(child))]

        # Try a prepositional phrase
        if not ret['right']:
            for child in tok.children:
                if child.dep_ == 'prep':
                    ret['right'] = sorted(list(set(find_all_pobj(child))))

        if not ret['right']:
            for child in tok.children:
                if child.dep_ in ['attr', 'xcomp', 'ccomp']:
                    ret['right'] = [' '.join([child.text] + find_all_singletons(child) + find_all_modifiers(child))]

        for k in ['left', 'right']:
            ret[k] = [w.lower() for w in ret[k]]

        # conjunctions and adverbial clause modifiers
        for child in tok.children:
            if child.dep_ in ['conj', 'advcl']:
                lefts = list(child.lefts)
                found = ' '.join(ret['left'])
                if lefts:
                    conj = doc[lefts[0].i:]
                else:
                    conj = doc[child.i:]

                if not any(pos in [t.pos_ for t in conj] for pos in ['AUX', 'VERB']):
                    found = found + f" {ret['rel']}"

                conj_phrase = nlp(f'{found} ' + ' '.join([t.text for t in conj]))
                clauses += get_relationships(conj_phrase, depth=depth + 1)

        # Only include a clause if it has a left, rel, and right.
        if all(ret.values()) and ret not in clauses:
            clauses.insert(0, ret)

    return clauses

def to_archetype(doc):
    '''
    Convert a string or doc to a string with all proper nouns replaced by an archetype.
    '''
    if not isinstance(doc, spacy.tokens.doc.Doc):
        doc = nlp(doc)

    ret = []

    subs = dict(zip(list(dict.fromkeys([str(e) for e in doc.ents])), archetypes))
    subs = dict(zip(list(dict.fromkeys([str(e) for e in doc if e.pos_ == 'PROPN' ])), archetypes))

    if not subs:
        return str(doc)

    for tok in doc:
        if tok.text in subs:
            ret.append(subs[tok.text])
        else:
            if tok.dep_ == 'punct':
                ret[-1] = ret[-1] + tok.text
            else:
                ret.append(tok.text)

    return ' '.join(ret)
