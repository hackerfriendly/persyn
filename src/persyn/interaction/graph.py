''' graph.py: knowledge graph support by Neo4j '''
# pylint: disable=no-member, abstract-method

import re

from dataclasses import dataclass
from typing import Any, List, Optional

from neomodel import DateTimeProperty, StringProperty, UniqueIdProperty, IntegerProperty, RelationshipTo, StructuredRel, Q
from neomodel import config as neomodel_config
from neomodel import db as neomodel_db
from neomodel.contrib import SemiStructuredNode

from persyn.utils.color_logging import log

from persyn.utils.config import PersynConfig


# believes desires owns supports criticizes trusts knows ignores challenges values respects shares doubts rejects learns hopes contains

# Neomodel (Neo4j) graph classes
class PredicateRel(StructuredRel):
    ''' Predicate relationship: s P o '''
    weight = IntegerProperty(default=0)
    verb = StringProperty(required=True)

class Entity(SemiStructuredNode):
    ''' Superclass for everything '''
    name = StringProperty(required=True)
    bot_id = StringProperty(required=True)
    entity_id = UniqueIdProperty()
    link = RelationshipTo('Entity', 'LINK', model=PredicateRel)

class Person(Entity):
    ''' Something you can have a conversation with '''
    created = DateTimeProperty(default_now=True)

class Place(Entity):
    ''' Somewhere to be '''
    created = DateTimeProperty(default_now=True)

class Thing(Entity):
    ''' Any old thing '''
    created = DateTimeProperty(default_now=True)

@dataclass
class KnowledgeGraph:
    ''' Knowledge graph support by Neo4j '''
    config: PersynConfig

    def __post_init__(self):
        if hasattr(self.config.memory, 'neo4j'):
            neomodel_config.DATABASE_URL = self.config.memory.neo4j.url
        self.bot_id = self.config.id.guid
        self.name = self.config.id.name


    def delete_all_nodes(self, confirm=False) -> bool:
        ''' Delete all graph nodes for this bot '''
        if confirm is not True:
            return False

        for node in Entity.nodes.filter(Q(bot_id=self.bot_id)):
            node.delete()

        return True

    def person(self, name) -> Person:
        ''' Return a Person node with the given name '''
        return Person(name=name, bot_id=self.bot_id).save()

    def place(self, name) -> Place:
        ''' Return a Place node with the given name '''
        return Place(name=name, bot_id=self.bot_id).save()

    def thing(self, name) -> Thing:
        ''' Return a Thing node with the given name '''
        return Thing(name=name, bot_id=self.bot_id).save()

    def fetch_all_nodes(self, node_type=None) -> List[Any]:
        ''' Return all graph nodes for this bot '''
        if node_type is None:
            return Entity.nodes.filter(Q(bot_id=self.bot_id))
        if node_type == 'Person':
            return Person.nodes.filter(Q(bot_id=self.bot_id))
        if node_type == 'Place':
            return Place.nodes.filter(Q(bot_id=self.bot_id))
        if node_type == 'Thing':
            return Thing.nodes.filter(Q(bot_id=self.bot_id))
        raise RuntimeError(f'Invalid node_type: {node_type}')

    def find_node(self, name, node_type=None) -> List[Any]:
        ''' Return all nodes with the given name'''
        if node_type is None:
            return Entity.nodes.filter(Q(bot_id=self.bot_id), Q(name=name))
        if node_type == 'Person':
            return Person.nodes.filter(Q(bot_id=self.bot_id), Q(name=name))
        if node_type == 'Place':
            return Place.nodes.filter(Q(bot_id=self.bot_id), Q(name=name))
        if node_type == 'Thing':
            return Thing.nodes.filter(Q(bot_id=self.bot_id), Q(name=name))
        raise RuntimeError(f'Invalid node_type: {node_type}')

    @staticmethod
    def safe_name(name) -> str:
        ''' Return name sanitized as alphanumeric, space, or comma only, max 64 characters. '''
        return re.sub(r"[^a-zA-Z0-9, ]+", '', name.strip())[:64]

    def shortest_path(self, src, dest, src_type=None, dest_type=None, min_distance=0):
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
        WHERE length(p) > {min_distance}
        RETURN p
        """

        paths = neomodel_db.cypher_query(query)[0]
        ret = []
        if not paths:
            return ret

        for rel in paths[0][0].relationships:
            ret.append((rel.start_node.get('name'), rel.get('verb'), rel.end_node.get('name')))

        return ret

    def triples_to_kg(self, triples: List[tuple[str, str, str]], types: Optional[dict[str, str]]=None):
        '''
        Convert subject, predicate, object triples into a Neo4j graph.
        If types is provided, map {name: type} for the subjects and objects.
        '''
        if not hasattr(self.config.memory, 'neo4j'):
            log.error('፨ No graph server defined, cannot call triples_to_kg()')
            return

        if types is None:
            types = {}

        for k, v in types.items():
            if v == 'Person':
                types[k] = Person
            elif v == 'Place':
                types[k] = Place
            elif v == 'Thing':
                types[k] = Thing
            else:
                log.error('፨ Invalid type:', k)
                return

        with neomodel_db.transaction:
            for triple in triples:
                (subj, pred, obj) = triple

                if subj not in types:
                    types[subj] = Entity

                if obj not in types:
                    types[obj] = Entity

                try:
                    the_subject = types[subj].nodes.get(name=subj, bot_id=self.bot_id)
                except types[subj].DoesNotExist:
                    the_subject = types[subj](name=subj, bot_id=self.bot_id).save()

                try:
                    the_object = types[obj].nodes.get(name=obj, bot_id=self.bot_id)
                except types[obj].DoesNotExist:
                    the_object = types[obj](name=obj, bot_id=self.bot_id).save()

                the_subject.link.connect(the_object, {'verb': pred})
