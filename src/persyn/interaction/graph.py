''' graph.py: knowledge graph support by Neo4j '''
from dataclasses import dataclass
from neomodel import DateTimeProperty, StringProperty, UniqueIdProperty, FloatProperty, IntegerProperty, RelationshipTo, StructuredRel, Q
from neomodel import config as neomodel_config
from neomodel import db as neomodel_db
from neomodel.contrib import SemiStructuredNode

from persyn.utils.config import PersynConfig # pylint: disable=no-member

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

@dataclass
class KnowledgeGraph:
    config: PersynConfig

    def __post_init__(self):
        if hasattr(self.config.memory, 'neo4j'):
            neomodel_config.DATABASE_URL = self.config.memory.neo4j.url

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
        if not hasattr(self.config.memory, 'neo4j'):
            log.error('፨ No graph server defined, cannot call triples_to_kg()')
            return

        speaker_names = {p.name for p in Person.nodes.all() if p.bot_id == self.bot_id}
        thing_names = set()
        speakers = {}

        for triple in triples:
            (s, _, o) = triple

            if s not in speaker_names:
                thing_names.add(s)

            if o not in speaker_names:
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
