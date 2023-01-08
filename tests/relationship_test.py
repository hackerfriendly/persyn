'''
relationship tests
'''
# pylint: disable=import-error, wrong-import-position, invalid-name, line-too-long

import random

from interaction.relationships import Relationships

# Bot config
from utils.config import load_config

persyn_config = load_config()
relobj = Relationships(persyn_config)

test_cases_simple = {
    "A tripedal woman is quite unique, even in the art world.":
        [{'left': ['a tripedal woman'], 'rel': 'be', 'right': ['unique quite']}],
    "Anna agree with you that it doesn't sound particularly fun.":
        [{'left': ['anna'], 'rel': 'agree', 'right': ['you']}],
    "Anna and Ricky and their friend Jim's cousin's butler, Phil, discussed the work of Erving Goffman and the commonalities between various activities.":
        [{'left': ['anna', 'phil', 'ricky'], 'rel': 'discuss', 'right': ['the work']}],
    "Anna and Ricky and their friend's cousin's dog Phil discussed the work of Erving Goffman and the commonalities between various activities.":
        [{'left': ['anna', 'phil', 'ricky'], 'rel': 'discuss', 'right': ['the work']}],
    "Anna recalls was thinking about Bill, the tennis guy.":
        [{'left': ['anna'], 'rel': 'recall', 'right': ['thinking']}],
    "Even when other kids his age had left to play professional football or basketball, Bill stayed dedicated to his passion for tennis and continued to practice hard every day.":
        [{'left': ['bill'], 'rel': 'stay', 'right': ['dedicated']}, {'left': ['bill'], 'rel': 'continue', 'right': ['practice']}],
    "He didn't start playing at the age of 8 but quickly became known as one of the best players in town.":
        [{'left': ['he'], 'rel': 'not start', 'right': ['playing']}],
    "He started playing at the age of 8.":
        [{'left': ['he'], 'rel': 'start', 'right': ['playing at the age of 8']}],
    "He was a programmer trying to solve an issue with his computer, but he wasn't sure how.":
        [{'left': ['he'], 'rel': 'be', 'right': ['trying']}, {'left': ['he'], 'rel': 'not be how', 'right': ['sure']}],
    "Hi Anna, did you notice that one of your women in the picture is tripedal?":
        [{'left': ['you'], 'rel': 'notice', 'right': ['is tripedal']}, {'left': ['is tripedal'], 'rel': 'punct', 'right': ['?']}],
    "In desperation, he took it apart and managed to fix it himself.":
        [{'left': ['he'], 'rel': 'take apart', 'right': ['it']}, {'left': ['he'], 'rel': 'manage', 'right': ['fix']}],
    "It looks like she has a lot of character!":
        [{'left': ['she'], 'rel': 'have', 'right': ['a lot of character']}, {'left': ['a lot of character'], 'rel': 'punct', 'right': ['!']}],
    "It's fascinating to think about the possibilities!":
        [{'left': ['it'], 'rel': 'be', 'right': ['fascinating']}, {'left': ['fascinating'], 'rel': 'punct', 'right': ['!']}],
    "Rob was a programmer trying to solve an issue with his computer, but he wasn't sure how.":
        [{'left': ['rob'], 'rel': 'be', 'right': ['trying']}, {'left': ['he'], 'rel': 'not be how', 'right': ['sure']}],
    "She looks confident and composed, but also a bit mischievous.":
        [{'left': ['she'], 'rel': 'look', 'right': ['confident']}, {'left': ['she'], 'rel': 'look', 'right': ['mischievous bit']}],
    "She wanted to pursue She dream of becoming a yoga instructor and found greater opportunities in other countries .":
        [{'left': ['she'], 'rel': 'want', 'right': ['pursue']}, {'left': ['she'], 'rel': 'find', 'right': ['greater opportunities in other countries']}],
    "That doesn't actually sound like fun, for the person stuck in VR with you.":
        [{'left': ['that'], 'rel': 'not sound actually', 'right': ['fun']}, {'left': ['the person'], 'rel': 'stick', 'right': ['you']}],
    relobj.to_archetype("Anna and Hackerfriendly discussed the concept of emotional intelligence and then Anna proposed exploring Erving Goffman's work and its potential implications."):
        [{'left': ['alice', 'bob'], 'rel': 'discuss', 'right': ['the concept of emotional intelligence']}, {'left': ['alice'], 'rel': 'propose then', 'right': ['exploring']}],
    relobj.to_archetype("Rob was a programmer trying to solve an issue with his computer, but he wasn't sure how."):
        [{'left': ['alice'], 'rel': 'be', 'right': ['trying']}, {'left': ['he'], 'rel': 'not be how', 'right': ['sure']}],

    # This one can be parsed "she | hold | it" or "it | hold | she", so skip for now.
    # "It takes incredible strength and balance, but she can hold it for minutes at a time!":
    #     [{'left': ['it'], 'rel': 'take', 'right': ['incredible strength']}, {'left': ['she'], 'rel': 'hold', 'right': ['it']}],
}

test_cases_propn = {
    "hackerfriendly was thinking about Bill, the tennis guy, and his buddy Charlie.":
        [{'left': ['hackerfriendly'], 'rel': 'think', 'right': ['bill', 'charlie', 'the tennis guy']}],
    "Hackerfriendly was thinking about Bill, the tennis guy.":
        [{'left': ['hackerfriendly'], 'rel': 'think', 'right': ['the tennis guy']}],
    "Alice was thinking about Bill the tennis guy, and his buddy Charlie.":
        [{'left': ['alice'], 'rel': 'think', 'right': ['charlie', 'the tennis guy']}],
}

def test_archetypes():
    ''' Archetype name substitution '''
    names = [
        "Emma", "Thomas", "David", "Lucas",
        "Jacob", "Alex", "Avery", "Aaron", "John",
        "Joshua", "Noah", "Eva", "Michael", "Isabella",
        "Lily", "Ryan", "Brian", "Bella", "Abigail",
        "Hannah", "Adam", "Olivia", "Julia", "Grace",
        "Claire", "Rob"
    ]

    assert len(names) == len(relobj.archetypes)

    random.shuffle(names)
    assert relobj.to_archetype(' '.join(names)) == ' '.join(relobj.archetypes)

    # When we run out of archetypes, stop substituting
    assert relobj.to_archetype(' '.join(names + ["Scrooge"])) == ' '.join(relobj.archetypes + ["Scrooge"])

    sent = "%s and %s went to the park with %s and %s."
    assert relobj.to_archetype(sent % tuple(names[:4])) == "Alice and Bob went to the park with Carol and Dave."

def test_sentences_simple():
    ''' Check relationships for known sentences '''
    for sent, result in test_cases_simple.items():
        print(sent)
        assert relobj.get_relationships(sent) == result

def test_sentences_propn():
    '''
    Check relationships for known sentences with ambiguous proper names.
    This requires custom ruler patterns to work.
    '''
    patterns = [[{"LOWER": "hackerfriendly"}]]
    attrs = {"TAG": "NNP", "POS": "PROPN", "DEP": "nsubj"}

    ruler = relobj.nlp.get_pipe("attribute_ruler")
    ruler.add(patterns=patterns, attrs=attrs)

    for sent, result in test_cases_propn.items():
        print(sent)
        assert relobj.get_relationships(sent) == result

def test_edgelist():
    '''
    Edgelist construction
    '''
    for sent, relationships in list(test_cases_simple.items()):
        assert relobj.relations_to_edgelist(relobj.get_relationships(sent)) == relobj.relations_to_edgelist(relationships)

def test_graph():
    '''
    Construct a relationship graph.
    '''
    for sent, relationships in list(test_cases_simple.items()):
        g1 = relobj.relations_to_graph(relobj.get_relationships(sent))
        g2 = relobj.relations_to_graph(relationships)

        assert relobj.graph_similarity(g1, g2) == 1.0

def test_get_relationship_graph():
    '''
    Verify get_relationship_graph()
    '''
    for text in list(['Rob and Anna had a discussion about philosophy.', 'Bill and Ted are excellent to each other.']):
        # get_relationship_graph() resolves coreferences and does optional archetype substitution
        G = relobj.relations_to_graph(relobj.get_relationships(relobj.referee(text)))
        Grg = relobj.get_relationship_graph(text, include_archetypes=False)

        assert relobj.edge_similarity(G, Grg) == 1.0
        assert relobj.node_similarity(G, Grg) == 1.0
        assert relobj.graph_similarity(G, Grg) == 1.0

        # additional nodes and edges
        Grga = relobj.get_relationship_graph(text, include_archetypes=True)
        assert relobj.node_similarity(G, Grga) < 1.0
        assert relobj.graph_similarity(G, Grga) < 1.0
        assert len(G.nodes()) < len(Grga.nodes())
        assert len(G.edges()) < len(Grga.edges())

def test_graph_similarity():
    ''' Use jaccard_similarity() to test the similarity of two graphs '''

    # no ZeroDivisionError
    assert relobj.jaccard_similarity([], []) == 1.0

    try:
        for sent, relationships in list(test_cases_simple.items()):
            g1 = relobj.relations_to_graph(relobj.get_relationships(sent))
            g2 = relobj.relations_to_graph(relationships)

            # identity
            assert relobj.jaccard_similarity(g1.nodes(), g2.nodes()) == 1.0
            assert relobj.jaccard_similarity(g1.edges(), g2.edges()) == 1.0

            # node + edge similarity
            assert relobj.graph_similarity(g1, g1) == 1.0
            assert relobj.graph_similarity(g2, g2) == 1.0
            assert relobj.graph_similarity(g1, g2) == 1.0

            # removing a node also impacts edges
            g1.remove_node(list(g1.nodes())[0])
            assert relobj.jaccard_similarity(g1.nodes(), g2.nodes()) < 1.0
            assert relobj.jaccard_similarity(g1.edges(), g2.edges()) < 1.0
            assert relobj.graph_similarity(g1, g2) < 1.0

            g2.remove_node(list(g2.nodes())[0])

            # adding an edge also impacts nodes, so reuse an existing node
            g1.add_edge(list(g1.nodes())[0], list(g1.nodes())[0], edge='agree')

            # nodes are now identical
            assert relobj.jaccard_similarity(g1.nodes(), g2.nodes()) == 1.0

            # graphs are not
            assert relobj.graph_similarity(g1, g2) < 1.0

            # don't count the edges
            assert relobj.graph_similarity(g1, g2, edge_bias=0) == 1.0

            # higher edge bias gives more weight to edge matches
            g3 = relobj.relations_to_graph(relationships)
            g4 = relobj.relations_to_graph(relationships)
            assert relobj.graph_similarity(g3, g4) == 1.0

            g4.add_node('alien')
            almost = relobj.graph_similarity(g3, g4)
            assert almost < 1.0
            assert relobj.graph_similarity(g3, g4, edge_bias=3) > almost
            assert relobj.graph_similarity(g3, g4, edge_bias=5) > relobj.graph_similarity(g3, g4, edge_bias=3)

    except AssertionError as err:
        print("Nodes:", g1.nodes(), g2.nodes())
        print("Edges:", g1.edges(data=True), g2.edges(data=True))
        raise err
