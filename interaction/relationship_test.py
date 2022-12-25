'''
relationship tests
'''
# pylint: disable=import-error, wrong-import-position, invalid-name, line-too-long

import random

import networkx as nx

from relationships import get_relationships, to_archetype, jaccard_similarity, relations_to_graph, nlp, nlp_merged, archetypes

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
        [{'left': ['you'], 'rel': 'notice', 'right': ['is tripedal']}],
    "In desperation, he took it apart and managed to fix it himself.":
        [{'left': ['he'], 'rel': 'take apart', 'right': ['it']}, {'left': ['he'], 'rel': 'manage', 'right': ['fix']}],
    "It looks like she has a lot of character.":
        [{'left': ['she'], 'rel': 'have', 'right': ['a lot of character']}],
    "It's fascinating to think about the possibilities!":
        [{'left': ['it'], 'rel': 'be', 'right': ['fascinating']}],
    "Rob was a programmer trying to solve an issue with his computer, but he wasn't sure how.":
        [{'left': ['rob'], 'rel': 'be', 'right': ['trying']}, {'left': ['he'], 'rel': 'not be how', 'right': ['sure']}],
    "She looks confident and composed, but also a bit mischievous.":
        [{'left': ['she'], 'rel': 'look', 'right': ['confident']}, {'left': ['she'], 'rel': 'look', 'right': ['mischievous bit']}],
    "She wanted to pursue She dream of becoming a yoga instructor and found greater opportunities in other countries .":
        [{'left': ['she'], 'rel': 'want', 'right': ['pursue']}, {'left': ['she'], 'rel': 'find', 'right': ['greater opportunities in other countries']}],
    "That doesn't actually sound like fun, for the person stuck in VR with you.":
        [{'left': ['that'], 'rel': 'not sound actually', 'right': ['fun']}, {'left': ['the person'], 'rel': 'stick', 'right': ['you']}],
    to_archetype("Anna and Hackerfriendly discussed the concept of emotional intelligence and then Anna proposed exploring Erving Goffman's work and its potential implications."):
        [{'left': ['alice', 'bob'], 'rel': 'discuss', 'right': ['the concept of emotional intelligence']}, {'left': ['alice'], 'rel': 'propose then', 'right': ['exploring']}],
    to_archetype("Rob was a programmer trying to solve an issue with his computer, but he wasn't sure how."):
        [{'left': ['alice'], 'rel': 'be', 'right': ['trying']}, {'left': ['he'], 'rel': 'not be how', 'right': ['sure']}],

    # This one can be parsed "she | hold | it" or "it | hold | she", so skip for now.
    # "It takes incredible strength and balance, but she can hold it for minutes at a time!":
    #     [{'left': ['it'], 'rel': 'take', 'right': ['incredible strength']}, {'left': ['she'], 'rel': 'hold', 'right': ['it']}],
}

test_cases_propn = {
    "hackerfriendly was thinking about Bill, the tennis guy, and his buddy Charlie.": [{'left': ['hackerfriendly'], 'rel': 'think', 'right': ['bill', 'charlie', 'the tennis guy']}],
    "Hackerfriendly was thinking about Bill, the tennis guy.": [{'left': ['hackerfriendly'], 'rel': 'think', 'right': ['the tennis guy']}],
    "Alice was thinking about Bill the tennis guy, and his buddy Charlie.": [{'left': ['alice'], 'rel': 'think', 'right': ['charlie', 'the tennis guy']}],
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

    assert len(names) == len(archetypes)

    random.shuffle(names)
    assert to_archetype(' '.join(names)) == ' '.join(archetypes)

    sent = "%s and %s went to the park with %s and %s."
    assert to_archetype(sent % tuple(names[:4])) == "Alice and Bob went to the park with Carol and Dave."

def test_sentences_simple():
    ''' Check relationships for known sentences '''
    for sent, result in test_cases_simple.items():
        print(sent)
        assert get_relationships(sent) == result

def test_sentences_propn():
    '''
    Check relationships for known sentences with ambiguous proper names.
    This requires custom ruler patterns to work.
    '''
    patterns = [[{"LOWER": "hackerfriendly"}]]
    attrs = {"TAG": "NNP", "POS": "PROPN", "DEP": "nsubj"}

    ruler = nlp.get_pipe("attribute_ruler")
    ruler.add(patterns=patterns, attrs=attrs)

    ruler = nlp_merged.get_pipe("attribute_ruler")
    ruler.add(patterns=patterns, attrs=attrs)

    for sent, result in test_cases_propn.items():
        print(sent)
        assert get_relationships(sent) == result

def test_graph():
    '''
    Construct a relationship graph.
    Sentence parsing may be probabilistic, so choose your sentences carefully.
    '''
    for sent, relationships in list(test_cases_simple.items()):
        graph = relations_to_graph([
            get_relationships(sent)
        ])

        nodes = set()
        edges = []
        for rel in relationships:
            left = ' '.join(rel['left'])
            right = ' '.join(rel['right'])

            nodes.add(left)
            nodes.add(right)
            edges.append({"edge": rel['rel'], "source": left, "target": right})

        nld = nx.node_link_data(graph)
        graph_nodes = [n['id'] for n in nld['nodes']]

        assert sorted(graph_nodes) == sorted(list(nodes))
        assert nld['links'] == edges
