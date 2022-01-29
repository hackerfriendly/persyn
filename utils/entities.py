''' entities.py: utility classes for identifying arbitrary entities. '''
import uuid

class EntityMapper():
    ''' Generate or look up distinct IDs for a given name in a context for a bot '''
    def __init__(
        self,
        bot_id=None,
    ):
        self.namespace = uuid.UUID(bot_id)

    def name_to_id(self, service, channel, name):
        ''' One distinct UUID per bot_id + service + channel + name '''
        key = f"{service}|{channel}|{name}".strip()
        return str(uuid.uuid5(self.namespace, key))

    def id_to_name(self, key):
        ''' look up key in elasticsearch '''
        raise NotImplementedError("id_to_name()")
