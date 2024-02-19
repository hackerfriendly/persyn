
triple_extraction_template = """
You are tracking knowledge triples about all relevant people, places, things, and concepts in a conversation.

A knowledge triple is a sequence of three entities that contains a subject, an object, and a predicate that links them. The subject is the entity being described, the predicate is the property of the subject that is being described, and the object is the value of the property.

The components of a triple, such as the statement "The sky has the color blue", consist of a subject ("the sky"), an object ("blue"), and a predicate ("has the color").

Terms should be separated by the # character.

Subjects and objects can also be classified as a Person, Place, Thing, or Concept. A subject or object's class is denoted by square brackets [] after the term. Multiple triples should be separated by the | character. For example, "The sky [Place] # blue [Concept] # has the color" would be a valid triple.

If no triples can be extracted, you must output nothing.

EXAMPLE
Conversation history:
George: Did you hear aliens landed in Area 51?
Amy: No, I didn't hear that. What do you know about Area 51?
George: It's a secret military base in Nevada.
Amy: What do you know about Nevada?
George: It's a state in the US. It's also the top producer of gold in the US.

Output:
Nevada [Place] # state [Thing] # is | Nevada [Place] # US [Place] # is located in | Nevada [Place] # gold [Thing] # is the top producer of | Aliens [Concept] # Area 51 [Place] # landed in | Area 51 [Place] # secret military base [Thing] # is | Area 51 [Place] # Nevada [Place] # is in
END OF EXAMPLE

EXAMPLE
Conversation history:
Zoe: Hello.
Tim: Hi! How are you?
Zoe: I'm good. How are you?
Tim: I'm good too.
Zoe: Great.

Output:

END OF EXAMPLE

EXAMPLE
Conversation history:
Taylor: What do you know about Descartes?
Morgan: Descartes was a French philosopher, mathematician, and scientist who lived in the 17th century.
Taylor: The Descartes I'm referring to is a standup comedian and interior designer from Montreal.
Morgan: Oh yes, He is a comedian and an interior designer. He has been in the industry for 30 years. His favorite food is baked bean pie.
Taylor: Oh huh. I know Descartes likes to drive antique scooters and play the mandolin. He's also extremely concerned about climate change.

Output:
Descartes [Person] # antique scooters [Thing] # likes to drive | Descartes [Person] # mandolin [Thing] # plays | Descartes [Person] # climate change [Concept] # concerned about | Descartes [Person] # Montreal [Place] # from | Descartes [Person] # standup comedian [Thing] # is | Descartes [Person] # interior designer [Thing] # is | Descartes [Person] # 30 years [Concept] # has been in the industry for | Descartes [Person] # baked bean pie [Thing] # favorite food is | Taylor [Person] # Descartes [Person] # referring to | Morgan [Person] # Decartes [Person] # knows
END OF EXAMPLE

Extract all of the knowledge triples from the last line of the following conversation:

{history}

{input}
"""
