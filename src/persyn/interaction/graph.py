''' graph.py: knowledge graph support by Redis '''

import uuid
import datetime

from redis_om import HashModel

# https://docs.redis.com/latest/modules/redisgraph/redisgraph-quickstart/

from redis.commands.graph.edge import Edge
from redis.commands.graph.node import Node

# Connect to a database
r = redis.Redis(host="tachikoma0.s9", port="6379")  # , password="<password>")

# Define a graph called SocialMedia
social_graph = r.graph(f"{uuid.uuid4()}:convo_id")

# Create nodes that represent users
users = {"Alex": Node(label="Person", properties={"name": "Alex", "age": 35}),
         "Jun": Node(label="Person", properties={"name": "Jun", "age": 33}),
         "Taylor": Node(label="Person", properties={"name": "Taylor", "age": 28}),
         "Noor": Node(label="Person", properties={"name": "Noor", "age": 41})}

# Add users to the graph as nodes
for key in users.keys():
    social_graph.add_node(users[key])

# Add relationships between user nodes
social_graph.add_edge(Edge(users["Alex"], "friends", users["Jun"]))
# Make the relationship bidirectional
social_graph.add_edge(Edge(users["Jun"], "friends", users["Alex"]))

social_graph.add_edge(Edge(users["Jun"], "friends", users["Taylor"]))
social_graph.add_edge(Edge(users["Taylor"], "friends", users["Jun"]))

social_graph.add_edge(Edge(users["Jun"], "friends", users["Noor"]))
social_graph.add_edge(Edge(users["Noor"], "friends", users["Jun"]))

social_graph.add_edge(Edge(users["Alex"], "friends", users["Noor"]))
social_graph.add_edge(Edge(users["Noor"], "friends", users["Alex"]))

# Create the graph in the database
social_graph.commit()

# Query the graph to find out how many friends Alex has
result1 = social_graph.query("MATCH (p1:Person {name: 'Alex'})-[:friends]->(p2:Person) RETURN count(p2)")
print("Alex's original friend count:", result1.result_set)

# Delete a relationship without deleting any user nodes
social_graph.query("MATCH (:Person {name: 'Alex'})<-[f:friends]->(:Person {name: 'Jun'}) DELETE f")

# Query the graph again to see Alex's updated friend count
result2 = social_graph.query("MATCH (p1:Person {name: 'Alex'})-[:friends]->(p2:Person) RETURN count(p2)")
print("Alex's updated friend count:", result2.result_set)



# Neomodel (Neo4j) graph classes
class Entity(SemiStructuredNode):
    ''' Superclass for everything '''
    name = StringProperty(required=True)
    bot_id = StringProperty(required=True)


class PredicateRel(StructuredRel):
    ''' Predicate relationship: s P o '''
    weight = IntegerProperty(default=0)
    verb = StringProperty(required=True)


class Person(Entity):
    ''' Something you can have a conversation with '''
    created = DateTimeProperty(default_now=True)
    last_contact = DateTimeProperty(default_now=True)
    entity_id = UniqueIdProperty()
    speaker_id = UniqueIdProperty()
    link = RelationshipTo('Entity', 'LINK', model=PredicateRel)


class Thing(Entity):
    ''' Any old thing '''
    created = DateTimeProperty(default_now=True)
    link = RelationshipTo('Entity', 'LINK', model=PredicateRel)


class Human(Person):
    ''' Flesh and blood '''
    last_contact = DateTimeProperty(default_now=True)
    trust = FloatProperty(default=0.0)


class Bot(Person):
    ''' Silicon and electricity '''
    service = StringProperty(required=True)
    channel = StringProperty(required=True)
    trust = FloatProperty(default=0.0)


class GroK():
    '''
    The Graph of Knowledge
    grŏk
    transitive verb
    1. To understand profoundly through intuition or empathy.
    '''

    def __init__(self, persyn_config):
        self.persyn_config = persyn_config
        self.bot_name = persyn_config.id.name
        self.bot_id = uuid.UUID(persyn_config.id.guid)

    def delete_all_nodes(self, confirm=False):
        ''' Delete all graph nodes for this bot '''
        if confirm is not True:
            return False

        for node in Entity.nodes.filter(Q(bot_id=self.bot_id)):
            node.delete()

        return True

    def fetch_all_nodes(self, node_type=None):
        ''' Return all graph nodes for this bot '''
        if node_type is None:
            return Entity.nodes.filter(Q(bot_id=self.bot_id))
        if node_type == 'person':
            return Person.nodes.filter(Q(bot_id=self.bot_id))
        if node_type == 'thing':
            return Thing.nodes.filter(Q(bot_id=self.bot_id))
        raise RuntimeError(f'Invalid node_type: {node_type}')

    def find_node(self, name, node_type=None):
        ''' Return all nodes with the given name'''
        if node_type is None:
            return Entity.nodes.filter(Q(bot_id=self.bot_id), Q(name=name))
        if node_type == 'person':
            return Person.nodes.filter(Q(bot_id=self.bot_id), Q(name=name))
        if node_type == 'thing':
            return Thing.nodes.filter(Q(bot_id=self.bot_id), Q(name=name))
        raise RuntimeError(f'Invalid node_type: {node_type}')

    @staticmethod
    def safe_name(name):  # TODO: unify this with gpt.py
        ''' Return name sanitized as alphanumeric, space, or comma only, max 64 characters. '''
        return re.sub(r"[^a-zA-Z0-9, ]+", '', name.strip())[:64]

    def shortest_path(self, src, dest, src_type=None, dest_type=None):
        '''
        Find the shortest path between two nodes, if any.
        If src_type or dest_type are specified, constrain the search to nodes of that type.
        Returns a list of triples (string names of nodes and edges) encountered along the path.
        '''
        safe_src = self.safe_name(src)
        safe_dest = self.safe_name(dest)

        if safe_src == safe_dest:
            return []

        query = f"""
        MATCH
        (a{':'+src_type if src_type else ''} {{name: '{safe_src}', bot_id: '{self.bot_id}'}}),
        (b{':'+dest_type if dest_type else ''} {{name: '{safe_dest}', bot_id: '{self.bot_id}'}}),
        p = shortestPath((a)-[*]-(b))
        WITH p
        WHERE length(p) > 1
        RETURN p
        """
        paths = neomodel_db.cypher_query(query)[0]
        ret = []
        if not paths:
            return ret

        for r in paths[0][0].relationships:
            ret.append((r.start_node.get('name'), r.get('verb'), r.end_node.get('name')))

        return ret

    def triples_to_kg(self, triples):
        '''
        Convert subject, predicate, object triples into a Neo4j graph.
        '''
        if not hasattr(self.persyn_config.memory, 'neo4j'):
            log.error('፨ No graph server defined, cannot call triples_to_kg()')
            return

        speaker_names = set()
        thing_names = set()
        speakers = {}

        for triple in triples:
            (s, _, o) = triple

            if s not in thing_names and self.lookup_speaker_name(s):
                speaker_names.add(s)
            else:
                thing_names.add(s)

            if o not in thing_names and self.lookup_speaker_name(o):
                speaker_names.add(o)
            else:
                thing_names.add(o)

        with neomodel_db.transaction:
            for name in speaker_names:
                try:
                    speakers[name] = Person.nodes.get(name=name, bot_id=self.bot_id)  # pylint: disable=no-member
                # If they don't yet exist in the graph, make a new node
                except Person.DoesNotExist:  # pylint: disable=no-member
                    speakers[name] = Person(name=name, bot_id=self.bot_id).save()

            things = {}
            for t in Thing.get_or_create(*[{'name': n, 'bot_id': self.bot_id} for n in list(thing_names) if n not in speakers]):
                things[t.name] = t

            for link in triples:
                if link[0] in speakers:
                    subj = speakers[link[0]]
                else:
                    subj = things[link[0]]

                pred = link[1]

                if link[2] in speakers:
                    obj = speakers[link[2]]
                else:
                    obj = things[link[2]]

                found = False
                for rel in subj.link.all_relationships(obj):
                    if rel.verb == pred:
                        found = True
                        rel.weight = rel.weight + 1
                        rel.save()

                if not found:
                    rel = subj.link.connect(obj, {'verb': pred})
