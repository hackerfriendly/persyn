import json

try:
    from cryptography.fernet import Fernet, InvalidToken
    has_crypto = True
except ImportError:
    has_crypto = False

class Serializer:
    def __init__(self):
        pass

    def load(self, blob, types):
        event = json.loads(blob)
        event_type = event.pop("type")
        if event_type not in types:
            return event_type, None
        cls = types[event_type]
        return event_type, cls(**event)

    def dump(self, obj):
        event_type = obj.__class__.__name__
        event = dict(obj)
        event["type"] = event_type
        return json.dumps(event)

class EncryptedSerializer(Serializer):
    def __init__(self, shared_key):
        if not has_crypto:
            raise RuntimeError("Encryption requested but cryptography module isn't installed")
        self.fernet = Fernet(shared_key)

    def load(self, blob, types):
        try:
            plaintext = self.fernet.decrypt(blob).decode("utf-8")
        except InvalidToken as e:
            return e, None
        return super().load(plaintext, types)

    def dump(self, obj):
        plaintext = super().dump(obj)
        return self.fernet.encrypt(plaintext.encode("utf-8"))

if __name__ == "__main__":
    import sys
    if not has_crypto:
        raise RuntimeError("Encryption requested but cryptography module isn't installed")
    if sys.argv[1] == "generate":
        print(Fernet.generate_key())
