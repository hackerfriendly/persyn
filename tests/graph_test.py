import pytest
from persyn.utils.config import load_config
from persyn.interaction.graph import KnowledgeGraph, Entity, Person, Place, Thing

persyn_config = load_config()
kg = KnowledgeGraph(config=persyn_config)
kg.delete_all_nodes(confirm=True)

@pytest.fixture
def cleanup():
    kg.delete_all_nodes(confirm=True)

# Tests follow

def test_fetch_all_nodes(cleanup):
    assert len(kg.fetch_all_nodes()) == 0
    kg.person("Test Person")
    kg.place("Test Place")
    kg.thing("Test Thing")
    kg.concept("Test Concept")
    assert len(kg.fetch_all_nodes()) == 4

def test_fetch_all_nodes_invalid(cleanup):
    with pytest.raises(RuntimeError):
        kg.fetch_all_nodes('invalid')

def test_safe_name():
    assert kg.safe_name('Test Name!') == 'Test Name'


def test_add_and_delete_person(cleanup):
    person = Person(name="Test Person", bot_id=kg.bot_id).save()
    assert len(kg.fetch_all_nodes('Person')) == 1
    person2 = kg.person("Test Person2")
    assert len(kg.fetch_all_nodes('Person')) == 2
    person.delete()
    assert len(kg.fetch_all_nodes('Person')) == 1
    person2.delete()
    assert len(kg.fetch_all_nodes('Person')) == 0


def test_add_and_delete_place(cleanup):
    place = Place(name="Test Place", bot_id=kg.bot_id).save()
    assert len(kg.fetch_all_nodes('Place')) == 1
    place.delete()
    assert len(kg.fetch_all_nodes('Place')) == 0


def test_add_and_delete_thing(cleanup):
    thing = Thing(name="Test Thing", bot_id=kg.bot_id).save()
    assert len(kg.fetch_all_nodes('Thing')) == 1
    thing.delete()
    assert len(kg.fetch_all_nodes('Thing')) == 0


def test_find_node(cleanup):
    person = Person(name="Test Person", bot_id=kg.bot_id).save()
    found_node = kg.find_node("Test Person", "Person")
    assert found_node is not None
    assert len(found_node) == 1
    assert found_node[0].name == "Test Person"
    person.delete()


def test_shortest_path(cleanup):
    person1 = Person(name="Person1", bot_id=kg.bot_id).save()
    person2 = Person(name="Person2", bot_id=kg.bot_id).save()
    person1.link.connect(person2, {'verb': 'knows'})
    path = kg.shortest_path("Person1", "Person2")

    assert len(path) == 1
    assert path[0] == ("Person1", "knows", "Person2")
    person3 = Person(name="Person3", bot_id=kg.bot_id).save()
    person4 = Person(name="Person4", bot_id=kg.bot_id).save()

    person2.link.connect(person3, {'verb': 'knows'})
    person3.link.connect(person4, {'verb': 'knows'})
    path = kg.shortest_path(
        src="Person1",
        dest="Person4",
        src_type="Person",
        dest_type="Person"
    )
    assert len(path) == 3

def test_shortest_path_place_thing(cleanup):
    place1 = kg.place(name='Place1')
    thing1 = kg.thing(name="Thing1")
    place1.link.connect(thing1, {'verb': 'contains'})

    # Wrong type
    path = kg.shortest_path("Place1", "Thing1", "Place", "Person")
    assert len(path) == 0

    # Correct type
    path = kg.shortest_path("Place1", "Thing1", "Place", "Thing")
    assert len(path) == 1
    assert path[0] == ("Place1", "contains", "Thing1")
    place1.delete()
    thing1.delete()

def test_save_triples(cleanup):
    triples = [("Person1", "knows", "Person2"), ("Person2", "lives", "Place1")]

    # Default type mapping, everything is an Entity
    kg.save_triples(triples)
    assert len(kg.fetch_all_nodes()) == 3
    assert len(kg.fetch_all_nodes('Person')) == 0
    assert len(kg.fetch_all_nodes('Place')) == 0
    assert len(kg.shortest_path("Person1", "Place1")) == 2

    # Start over
    kg.delete_all_nodes(confirm=True)
    assert len(kg.fetch_all_nodes()) == 0

    # Use the correct types
    kg.save_triples(triples, {'Person1': 'Person', 'Person2': 'Person', 'Place1': 'Place'})

    assert len(kg.fetch_all_nodes('Person')) == 2
    assert len(kg.fetch_all_nodes('Place')) == 1
    assert len(kg.shortest_path("Person1", "Place1")) == 2

def test_cleanup(cleanup):
    assert True
