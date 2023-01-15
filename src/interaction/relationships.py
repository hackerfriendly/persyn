'''
relationships.py: extract a relationship graph from text
'''
# pylint: disable=c-extension-no-member, import-error, wrong-import-position, invalid-name, too-many-branches, unused-import
from operator import itemgetter

import spacy
import networkx as nx

from spacy.tokens import Doc

import coreferee

# Color logging
from utils.color_logging import ColorLog

log = ColorLog()

__all__ = [ 'Relationships' ]

class Relationships():
    ''' Container class for relationship graph operations. '''
    def __init__(self, persyn_config):
        self.persyn_config = persyn_config

        # Coreference resolved pipeline
        self.nlp = spacy.load(persyn_config.spacy.model)
        self.nlp.add_pipe('coreferee')
        self.nlp.add_pipe('sentencizer')

        # Merged pipeline
        self.nlp_merged = spacy.load(persyn_config.spacy.model)
        self.nlp_merged.add_pipe('merge_entities')
        self.nlp_merged.add_pipe('merge_noun_chunks')

        self.archetypes = [
            "Alice", "Bob", "Carol", "Dave", "Eve",
            "Frank", "Gavin", "Heidi", "Ivan", "Judy",
            "Kaitlin", "Larry", "Mia",
            "Natalie", "Oliver", "Peggy", "Quentin", "Rupert",
            "Sophia", "Trent", "Ursula", "Victor", "Wanda",
            "Xavier", "Yolanda", "Zahara"
        ]

    def referee(self, doc):
        '''
        Resolve coreferences in doc
        Returns a new doc with coreferences resolved
        '''
        if not isinstance(doc, spacy.tokens.doc.Doc):
            doc = self.nlp(doc)

        resolved = []
        for tok in doc:
            if doc._.coref_chains is None:
                resolved.append(tok.text)
                continue
            ccr = doc._.coref_chains.resolve(tok)
            if ccr is None:
                resolved.append(tok.text)
            else:
                for word in ccr:
                    resolved.append(word.text)

        return self.nlp(Doc(vocab=doc.vocab, words=resolved))

    @staticmethod
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

    def find_all_appos(self, tok, dep='conj'):
        ''' If tok.dep_ matches dep, return all children that are appositional modifiers '''
        ret = []
        for child in tok.children:
            if child.dep_ == dep:
                ret = [c.text for c in child.children if c.dep_ == 'appos']
                if not ret:
                    ret = [child.text] + self.find_all_appos(child)
        return ret

    def all_singletons(self, node):
        ''' Return True if all descendents are singletons, else False. '''
        if len(list(node.children)) > 1:
            return False

        for child in node.children:
            if not self.all_singletons(child):
                return False

        return True

    def find_all_singletons(self, tok):
        ''' Return a list of all descendants with only one child. '''
        if not list(tok.children):
            return []

        if not self.all_singletons(tok):
            return []

        ret = []
        for child in tok.children:
            ret = [child.text] + self.find_all_singletons(child)

        return ret

    def get_relationships(self, doc, depth=0):
        '''
        Find all relationships in doc
        Returns a list of { left: [], rel: "", right: [] } dicts
        '''
        if depth > 1:
            log.debug("💏 Maximum recursion depth reached.")
            return []

        clauses = []
        ret = {
            'left': [],
            'rel': None,
            'right': []
        }

        # Merge nouns
        doc = self.nlp_merged(str(doc))

        # Resolve coreferences
        doc = self.referee(doc)

        for sent in doc.sents:
            for tok in sent:

                # Find the ROOT
                if tok.dep_ != 'ROOT':
                    continue

                if tok.pos_ not in ['VERB', 'AUX']:
                    log.debug("💏 Root is not a verb, can't continue:", f"{tok} : {doc}")
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
                        subj = [child.text] + self.find_all_appos(child)
                        ret['left'] = sorted(list(set(subj)))

                    elif child.dep_ == 'dobj':
                        ret['right'] = [
                            ' '.join([child.text]
                            + self.find_all_singletons(child)
                            + self.find_all_modifiers(child))
                        ]

                # no dobj available, try something else
                if not ret['right']:
                    for child in tok.children:
                        # Try others
                        if child.dep_ == 'acomp':
                            ret['right'] = [
                                ' '.join([child.text]
                                + self.find_all_singletons(child)
                                + self.find_all_modifiers(child))
                            ]

                # Try a prepositional phrase
                if not ret['right']:
                    for child in tok.children:
                        if child.dep_ == 'prep':
                            ret['right'] = sorted(list(set(self.find_all_appos(child, dep='pobj'))))

                if not ret['right']:
                    for child in tok.children:
                        if child.dep_ in ['attr', 'xcomp', 'ccomp']:
                            ret['right'] = [
                                ' '.join([child.text]
                                + self.find_all_singletons(child)
                                + self.find_all_modifiers(child))
                            ]

                # lower everything
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

                        conj_phrase = self.nlp(f'{found} ' + ' '.join([t.text for t in conj]))
                        clauses += self.get_relationships(conj_phrase, depth=depth + 1)

                # Only include a clause if it has a left, rel, and right.
                if all(ret.values()) and ret not in clauses:
                    clauses.insert(0, ret)
                    # Add a link to terminating punctuation if it is a soldier or a hunchback
                    if sent[-1].dep_ == 'punct' and sent[-1].text in ['!', '?']:
                        clauses.append({'left': ret['right'], 'rel': 'punct', 'right': [sent[-1].text]})

        return clauses

    def to_archetype(self, doc):
        '''
        Convert a string or doc to a string with all proper nouns replaced by an archetype.

        If there are more proper nouns than archetypes, stop substituting at the last archetype.
        '''
        if not isinstance(doc, spacy.tokens.doc.Doc):
            doc = self.nlp(doc)

        ret = []

        subs = dict(
            zip(
                list(
                    dict.fromkeys([str(e) for e in doc if e.pos_ == 'PROPN'][:len(self.archetypes)])
                ),
                self.archetypes
            )
        )

        if not subs:
            return str(doc)

        for tok in doc:
            if tok.text in subs:
                ret.append(subs[tok.text])
            else:
                # Avoid , problems with spacing .
                if tok.dep_ == 'punct' and ret:
                    ret[-1] = ret[-1] + tok.text
                else:
                    ret.append(tok.text)

        return ' '.join(ret)

    @staticmethod
    def jaccard_similarity(g, h):
        '''
        Return the normalized similarity of two sets.
        https://en.wikipedia.org/wiki/Jaccard_index
        '''
        i = set(g).intersection(h)
        try:
            return len(i) / float(len(g) + len(h) - len(i))
        # Only possible if g and h are empty, and therefore identical
        except ZeroDivisionError:
            return 1.0

    def edge_similarity(self, g1, g2):
        '''
        Compute the Jaccard similarity of the edges of two graphs
        '''
        return self.jaccard_similarity(
            [str(e) for e in g1.edges(data=True)],
            [str(e) for e in g2.edges(data=True)]
        )

    def node_similarity(self, g1, g2):
        '''
        Compute the Jaccard similarity of the nodes of two graphs
        '''
        return self.jaccard_similarity(g1.nodes(), g2.nodes())

    def graph_similarity(self, g1, g2, edge_bias=0.5):
        '''
        Compute the total similarity of two graphs as a single normalized metric.
        Applies edge_bias to edge comparisons. 1.0 ignores nodes, 0.0 ignores edges.
        '''
        return (
                edge_bias * self.edge_similarity(g1, g2) +
                (1 - edge_bias) * self.node_similarity(g1, g2)
        )

    def relations_to_graph(self, relations, graph_type=nx.DiGraph):
        ''' Construct a graph from a list of relations '''
        return nx.from_edgelist(self.relations_to_edgelist(relations), create_using=graph_type)

    @staticmethod
    def relations_to_edgelist(relations):
        ''' Construct an edgelist from a list of relations '''
        ret = []
        for rel in relations:
            for left in rel['left']:
                for right in rel['right']:
                    ret.append((left, right, {'edge': rel['rel']}))
        return ret

    def get_relationship_graph(self, text, include_archetypes=True, graph_type=nx.DiGraph):
        '''
        Build a relationship graph from text:
        * Coreference resolution is run on text
        * Relationships are extracted from each sentence
        * A directed graph is made of those relationships

        If include_archetypes is True, also perform archetype substitution
        and include those nodes and edges.

        Returns an nx graph.
        '''
        edgelist = []
        for sent in self.nlp(self.referee(text)).sents:
            for rel in self.relations_to_edgelist(self.get_relationships(sent)):
                edgelist.append(rel)

        if include_archetypes:
            for sent in self.nlp(self.referee(self.to_archetype(text))).sents:
                for rel in self.relations_to_edgelist(self.get_relationships(sent)):
                    edgelist.append(rel)

        return nx.from_edgelist(edgelist, create_using=graph_type)

    @staticmethod
    def load_graph(hit):
        ''' Load an nx graph from an ES hit '''
        return nx.node_link_graph(hit['_source']['graph'])

    def best_match(self, G, hits, edge_bias=0.5):
        '''
        Return the best match to G from a list of potential matches,
        or None if there are no matches.
        '''
        ranked = self.ranked_matches(G, hits, edge_bias)
        if ranked:
            return ranked[0]
        return None

    def ranked_matches(self, G, hits, edge_bias=0.5):
        '''
        Rank all possible matches to G from a list of potential ES matches.

        Returns a list of matches and their scores, or [] if there are no matches
        '''
        scores = {}
        matches = {}
        for hit in hits:
            gh = self.load_graph(hit)

            # Skip graphs with zero matching edges
            if edge_bias and not self.edge_similarity(G, gh):
                continue

            score = round(self.graph_similarity(G, gh, edge_bias), 3)
            scores[hit['_id']] = score
            matches[hit['_id']] = hit

        if not scores:
            return []

        # Floats make terrible keys.
        sorted_scores = sorted(scores.items(), key=itemgetter(1), reverse=True)

        ranked = []
        for score in sorted_scores:
            ranked.append({"hit": matches[score[0]], "score": score[1]})

        log.warning("📈 Graph match scores:")
        for hit in ranked:
            log.warning(f"  {hit['score']}", hit['hit']['_source']['convo'][:100])

        return ranked

    @staticmethod
    def graph_to_json(G):
        ''' Convert an nx graph to node link data JSON '''
        return nx.node_link_data(G)
