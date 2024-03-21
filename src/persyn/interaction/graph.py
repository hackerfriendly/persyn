''' graph.py: knowledge graph support by Neo4j '''
# pylint: disable=no-member, abstract-method

import re

from dataclasses import dataclass
from typing import Any, List

from neomodel import DateTimeProperty, StringProperty, UniqueIdProperty, IntegerProperty, RelationshipTo, StructuredRel, Q
from neomodel import config as neomodel_config
from neomodel import db as neomodel_db
from neomodel.contrib import SemiStructuredNode

# Prompt completion
from persyn.interaction.completion import LanguageModel

from persyn.utils.color_logging import log

from persyn.utils.config import PersynConfig

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

class Concept(Entity):
    ''' Something less tangible '''
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
        self.lm = LanguageModel(self.config)  # pylint: disable=invalid-name

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

    def concept(self, name) -> Concept:
        ''' Return a Concept node with the given name '''
        return Concept(name=name, bot_id=self.bot_id).save()

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
        if node_type == 'Concept':
            return Concept.nodes.filter(Q(bot_id=self.bot_id))
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
        if node_type == 'Concept':
            return Concept.nodes.filter(Q(bot_id=self.bot_id), Q(name=name))
        raise RuntimeError(f'Invalid node_type: {node_type}')

    @staticmethod
    def safe_name(name: str) -> str:
        ''' Return name sanitized as alphanumeric, space, or comma only, max 64 characters. '''
        return re.sub(r"[^a-zA-Z0-9, ]+", '', name.strip())[:64].strip()

    def shortest_path(self, src, dest, src_type=None, dest_type=None) -> list[Any]:
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
        RETURN p
        """

        paths = neomodel_db.cypher_query(query)[0]
        ret = []
        if not paths:
            return ret

        for rel in paths[0][0].relationships:
            ret.append((rel.start_node.get('name'), rel.get('verb'), rel.end_node.get('name')))

        return ret

    def find_path(self, src, dest, src_type=None, dest_type=None, via=None, via_type=None) -> list[Any]:
        '''
        Find a path between two nodes, if any. If via is specified, require the path to pass through a node of that type.
        If src_type, dest_type, or via_type are specified, constrain the search to nodes of that type.
        Returns a list of triples (string names of nodes and edges) encountered along the path.
        '''
        safe_src = self.safe_name(src)
        safe_dest = self.safe_name(dest)

        if safe_src == safe_dest:
            return []

        if via is None:
            return self.shortest_path(safe_src, safe_dest, src_type, dest_type)

        safe_via = self.safe_name(via)

        a_to_b = self.shortest_path(safe_src, safe_via, src_type, via_type)
        b_to_c = self.shortest_path(safe_dest, safe_via, dest_type, via_type)

        if not a_to_b or not b_to_c:
            return []

        return a_to_b + b_to_c[::-1]

    def save_triples(self, triples: List[tuple[str, str, str]]):
        '''
        Convert subject, predicate, object triples into a Neo4j graph.
        Subject and object may include an optional [type].
        '''
        if not hasattr(self.config.memory, 'neo4j'):
            log.error('፨ No graph server defined, cannot call save_triples()')
            return

        types = {
            'Person': Person,
            'Place': Place,
            'Thing': Thing,
            'Concept': Concept
        }

        with neomodel_db.transaction:
            for triple in triples:
                (subj, pred, obj) = triple

                if '[' in subj:
                    match = re.search(r'(.*) ?\[(.*)\]', subj)
                    subj, subj_type = match.group(1), match.group(2)
                    if subj_type not in types:
                        log.warning(f'፨ Unknown subject type: {subj_type}, coercing to Concept')
                        subj_type = 'Concept'
                else:
                    subj_type = 'Concept'

                if '[' in obj:
                    match = re.search(r'(.*) ?\[(.*)\]', obj)
                    obj, obj_type = match.group(1), match.group(2)
                    if obj_type not in types:
                        log.warning(f'፨ Unknown object type: {obj_type}, coercing to Concept')
                        obj_type = 'Concept'
                else:
                    obj_type = 'Concept'

                subj = self.safe_name(subj)
                obj = self.safe_name(obj)

                try:
                    the_subject = types[subj_type].nodes.get(name=subj, bot_id=self.bot_id)
                except types[subj_type].DoesNotExist:
                    the_subject = types[subj_type](name=subj, bot_id=self.bot_id).save()

                try:
                    the_object = types[obj_type].nodes.get(name=obj, bot_id=self.bot_id)
                except types[obj_type].DoesNotExist:
                    the_object = types[obj_type](name=obj, bot_id=self.bot_id).save()

                # TODO: If the link already exists, increment the weight
                the_subject.link.connect(the_object, {'verb': pred})
