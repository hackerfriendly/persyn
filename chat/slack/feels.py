'''
feels.py: emoji and emotions
'''
import random

# emotion "analysis". This is just one big regex.
import text2emotion as te

# scikit-learn profanity filter (alt-profanity-check)
from profanity_check import predict_prob as profanity_prob

# flair sentiment
import flair

# Only load the model once on import. Uses GPU if available.
flair_sentiment = flair.models.TextClassifier.load('en-sentiment')

# Not actually "all" emoji, but all the emoji we can randomly respond with.
all_emoji = (
    ':bowtie:', ':smile:', ':simple_smile:', ':laughing:', ':blush:', ':smiley:', ':relaxed:',
    ':smirk:', ':heart_eyes:', ':kissing_heart:', ':kissing_closed_eyes:', ':flushed:',
    ':relieved:', ':satisfied:', ':grin:', ':wink:', ':stuck_out_tongue_winking_eye:',
    ':stuck_out_tongue_closed_eyes:', ':grinning:', ':kissing:', ':kissing_smiling_eyes:',
    ':stuck_out_tongue:', ':sleeping:', ':worried:', ':frowning:', ':anguished:',
    ':open_mouth:', ':grimacing:', ':confused:', ':hushed:', ':expressionless:', ':unamused:',
    ':sweat_smile:', ':sweat:', ':disappointed_relieved:', ':weary:', ':pensive:',
    ':disappointed:', ':confounded:', ':fearful:', ':cold_sweat:', ':persevere:', ':cry:',
    ':sob:', ':joy:', ':astonished:', ':scream:', ':tired_face:', ':angry:',
    ':rage:', ':triumph:', ':sleepy:', ':yum:', ':mask:', ':sunglasses:', ':dizzy_face:',
    ':imp:', ':smiling_imp:', ':neutral_face:', ':no_mouth:', ':innocent:', ':alien:',
    ':yellow_heart:', ':blue_heart:', ':purple_heart:', ':heart:', ':green_heart:',
    ':broken_heart:', ':heartbeat:', ':heartpulse:', ':two_hearts:', ':revolving_hearts:',
    ':cupid:', ':sparkling_heart:', ':sparkles:', ':star:', ':star2:', ':dizzy:', ':boom:',
    ':collision:', ':anger:', ':exclamation:', ':question:', ':grey_exclamation:',
    ':grey_question:', ':zzz:', ':dash:', ':sweat_drops:', ':notes:', ':musical_note:',
    ':fire:', ':shit:', ':+1:', ':-1:',
    ':ok_hand:', ':punch:', ':fist:', ':v:', ':wave:', ':hand:',
    ':raised_hand:', ':open_hands:', ':point_up:', ':point_down:', ':point_left:',
    ':point_right:', ':raised_hands:', ':pray:', ':point_up_2:', ':clap:', ':muscle:',
    ':the_horns:', ':middle_finger:'
)

# Negatory, good buddy
nope_emoji = ['-1', 'hankey', 'no_entry', 'no_entry_sign']

# Behold the emoji emotional spectrum
spectrum_emoji = [
    ':imp:', ':angry:', ':rage:', ':triumph:', ':scream:', ':tired_face:',
    ':sweat:', ':cold_sweat:', ':fearful:', ':sob:', ':weary:', ':cry:', ':mask:',
    ':confounded:', ':persevere:', ':unamused:', ':confused:', ':dizzy_face:',
    ':disappointed_relieved:', ':disappointed:', ':worried:', ':anguished:',
    ':frowning:', ':astonished:', ':flushed:', ':open_mouth:', ':hushed:',
    ':pensive:', ':expressionless:', ':neutral_face:', ':grimacing:',
    ':no_mouth:', ':kissing:', ':relieved:', ':smirk:', ':relaxed:',
    ':simple_smile:', ':blush:', ':wink:', ':sunglasses:', ':yum:',
    ':stuck_out_tongue:', ':stuck_out_tongue_closed_eyes:',
    ':stuck_out_tongue_winking_eye:', ':smiley:', ':smile:', ':laughing:',
    ':sweat_smile:', ':joy:', ':grin:'
]

# Helper map for text2emotion
emotion_map = {
    'Happy': 'happy',
    'Angry': 'angry',
    'Surprise': 'surprised',
    'Sad': 'sad',
    'Fear': 'afraid'
}

# Varying degrees of feels
degrees = (
    'hardly', 'barely', 'a little', 'kind of', 'sort of', 'slightly', 'somewhat',
    'relatively', 'to some degree', 'more or less', 'fairly', 'moderately', 'just about',
    'passably', 'tolerably', 'reasonably', 'largely', 'pretty', 'quite', 'bordering on',
    'almost', 'thoroughly', 'truly', 'significantly', 'very', 'wholly', 'altogether',
    'entirely', 'totally', 'utterly', 'positively', 'absolutely'
)

def random_emoji():
    ''' :wink: '''
    return random.choice(all_emoji)

def get_spectrum(score):
    ''' Translate a score from -1 to 1 into an emoji on the spectrum '''
    return spectrum_emoji[int(((score + 1) / 2) * (len(spectrum_emoji) - 1))]

def get_degree(score):
    ''' Turn a 0.0-1.0 score into a degree '''
    return degrees[int(score * (len(degrees) - 1))]

def get_feels(prompt):
    ''' How do we feel about this conversation? Ask text2emotion and return an object + text. '''
    emotions = te.get_emotion(prompt)
    phrase = []
    for emo in emotions.items():
        if emo[1] < 0.2:
            continue
        phrase.append(f"{get_degree(emo[1])} {emotion_map[emo[0]]}")

    if not phrase:
        return {"obj": emotions, "text": "nothing in particular"}

    if len(phrase) == 1:
        return {"obj": emotions, "text": phrase[0]}

    return {"obj": emotions, "text": ', '.join(phrase[:-1]) + f", and {phrase[-1]}"}

def rank_feels(some_feels):
    ''' Distill the feels obj into a float. 0 is neutral, range -1 to +1 '''
    score = 0.0
    emo = some_feels["obj"]
    for k in list(emo):
        # Fear is half positive
        if k == 'Fear':
            score = score + (emo[k] / 2.0)
        elif k in ('Happy', 'Surprise'):
            score = score + emo[k]
        else:
            score = score - emo[k]
    return score

def get_feels_score(prompt):
    ''' Turn text directly into a t2e ranked score '''
    return rank_feels(get_feels(prompt))

def get_profanity_score(prompt):
    ''' Profanity analysis with slkearn. Returns a float, -1.0 to 0 '''
    return -profanity_prob([prompt])[0]

def get_flair_score(prompt):
    ''' Run the flair sentiment prediction model. Returns a float, -1.0 to 1.0 '''
    sent = flair.data.Sentence(prompt)
    flair_sentiment.predict(sent)

    if sent.labels[0].value == 'NEGATIVE':
        return -sent.labels[0].score

    return sent.labels[0].score
