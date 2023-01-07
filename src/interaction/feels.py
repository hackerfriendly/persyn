'''
feels.py: emoji and emotions
'''
# pylint: disable=line-too-long

import random

# scikit-learn profanity filter (alt-profanity-check)
from profanity_check import predict_prob as profanity_prob

# edit distance
from Levenshtein import ratio

# flair sentiment
import flair

# spacy sentiment
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Only load the model once when we need it. Uses GPU if available.
flair_sentiment = None

# Not actually "all" emoji, but all the emoji we can randomly respond with.
reply_emoji = (
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

# Still not actually "all" emoji, but all the emoji we can try to match against.
all_emoji = (
  "+1", "-1", "100", "1234", "8ball", "a", "ab", "abacus",
  "abc", "abcd", "accept", "accordion", "adhesive_bandage", "admission_tickets", "adult", "aerial_tramway",
  "airplane", "airplane_arriving", "airplane_departure", "alarm_clock", "alembic", "alien", "ambulance", "amphora",
  "anatomical_heart", "anchor", "angel", "anger", "angry", "anguished", "ant", "apple",
  "aquarius", "aries", "arrow_backward", "arrow_double_down", "arrow_double_up", "arrow_down", "arrow_down_small", "arrow_forward",
  "arrow_heading_down", "arrow_heading_up", "arrow_left", "arrow_lower_left", "arrow_lower_right", "arrow_right", "arrow_right_hook", "arrow_up",
  "arrow_up_down", "arrow_up_small", "arrow_upper_left", "arrow_upper_right", "arrows_clockwise", "arrows_counterclockwise", "art", "articulated_lorry",
  "artist", "astonished", "astronaut", "athletic_shoe", "atm", "atom_symbol", "auto_rickshaw", "avocado",
  "axe", "b", "baby", "baby_bottle", "baby_chick", "baby_symbol", "back", "bacon",
  "badger", "badminton_racquet_and_shuttlecock", "bagel", "baggage_claim", "baguette_bread", "bald_man", "bald_person", "bald_woman",
  "ballet_shoes", "balloon", "ballot_box_with_ballot", "ballot_box_with_check", "bamboo", "banana", "bangbang", "banjo",
  "bank", "bar_chart", "barber", "barely_sunny", "baseball", "basket", "basketball", "bat",
  "bath", "bathtub", "battery", "beach_with_umbrella", "beans", "bear", "bearded_person", "beaver",
  "bed", "bee", "beer", "beers", "beetle", "beginner", "bell", "bell_pepper",
  "bellhop_bell", "bento", "beverage_box", "bicyclist", "bike", "bikini", "billed_cap", "biohazard_sign",
  "bird", "birthday", "bison", "biting_lip", "black_cat", "black_circle", "black_circle_for_record", "black_heart",
  "black_joker", "black_large_square", "black_left_pointing_double_triangle_with_vertical_bar", "black_medium_small_square", "black_medium_square", "black_nib", "black_right_pointing_double_triangle_with_vertical_bar", "black_right_pointing_triangle_with_double_vertical_bar",
  "black_small_square", "black_square_button", "black_square_for_stop", "blond-haired-man", "blond-haired-woman", "blossom", "blowfish", "blue_book",
  "blue_car", "blue_heart", "blueberries", "blush", "boar", "boat", "bomb", "bone",
  "book", "bookmark", "bookmark_tabs", "books", "boom", "boomerang", "boot", "bouquet",
  "bow", "bow_and_arrow", "bowl_with_spoon", "bowling", "boxing_glove", "boy", "brain", "bread",
  "breast-feeding", "bricks", "bride_with_veil", "bridge_at_night", "briefcase", "briefs", "broccoli", "broken_heart",
  "broom", "brown_heart", "bubble_tea", "bubbles", "bucket", "bug", "building_construction", "bulb",
  "bullettrain_front", "bullettrain_side", "burrito", "bus", "busstop", "bust_in_silhouette", "busts_in_silhouette", "butter",
  "butterfly", "cactus", "cake", "calendar", "call_me_hand", "calling", "camel", "camera",
  "camera_with_flash", "camping", "cancer", "candle", "candy", "canned_food", "canoe", "capital_abcd",
  "capricorn", "car", "card_file_box", "card_index", "card_index_dividers", "carousel_horse", "carpentry_saw", "carrot",
  "cat", "cat2", "cd", "chains", "chair", "champagne", "chart", "chart_with_downwards_trend",
  "chart_with_upwards_trend", "checkered_flag", "cheese_wedge", "cherries", "cherry_blossom", "chess_pawn", "chestnut", "chicken",
  "child", "children_crossing", "chipmunk", "chocolate_bar", "chopsticks", "christmas_tree", "church", "cinema",
  "circus_tent", "city_sunrise", "city_sunset", "cityscape", "cl", "clap", "clapper", "classical_building",
  "clinking_glasses", "clipboard", "clock12", "closed_book", "closed_lock_with_key", "closed_umbrella", "cloud", "clown_face",
  "clubs", "cn", "coat", "cockroach", "cocktail", "coconut", "coffee", "coffin",
  "coin", "cold_face", "cold_sweat", "comet", "compass", "compression", "computer", "confetti_ball",
  "confounded", "confused", "congratulations", "construction", "construction_worker", "control_knobs", "convenience_store", "cook",
  "cookie", "cool", "cop", "copyright", "coral", "corn", "couch_and_lamp", "couple_with_heart",
  "couplekiss", "cow", "cow2", "crab", "credit_card", "crescent_moon", "cricket", "cricket_bat_and_ball",
  "crocodile", "croissant", "crossed_fingers", "crossed_flags", "crossed_swords", "crown", "crutch", "cry",
  "crying_cat_face", "crystal_ball", "cucumber", "cup_with_straw", "cupcake", "cupid", "curling_stone", "curly_haired_man",
  "curly_haired_person", "curly_haired_woman", "curly_loop", "currency_exchange", "curry", "custard", "customs", "cut_of_meat",
  "cyclone", "dagger_knife", "dancer", "dancers", "dango", "dark_sunglasses", "dart", "dash",
  "date", "de", "deaf_man", "deaf_person", "deaf_woman", "deciduous_tree", "deer", "department_store",
  "derelict_house_building", "desert", "desert_island", "desktop_computer", "diamond_shape_with_a_dot_inside", "diamonds", "disappointed", "disappointed_relieved",
  "disguised_face", "diving_mask", "diya_lamp", "dizzy", "dizzy_face", "dna", "do_not_litter", "dodo",
  "dog", "dog2", "dollar", "dolls", "dolphin", "door", "dotted_line_face", "double_vertical_bar",
  "doughnut", "dove_of_peace", "dragon", "dragon_face", "dress", "dromedary_camel", "drooling_face", "drop_of_blood",
  "droplet", "drum_with_drumsticks", "duck", "dumpling", "dvd", "e-mail", "eagle", "ear",
  "ear_of_rice", "ear_with_hearing_aid", "earth_africa", "earth_americas", "earth_asia", "egg", "eggplant", "eight",
  "eight_pointed_black_star", "eight_spoked_asterisk", "eject", "electric_plug", "elephant", "elevator", "elf", "email",
  "empty_nest", "end", "envelope_with_arrow", "es", "euro", "european_castle", "european_post_office", "evergreen_tree",
  "exclamation", "exploding_head", "expressionless", "eye", "eye-in-speech-bubble", "eyeglasses", "eyes", "face_exhaling",
  "face_holding_back_tears", "face_in_clouds", "face_palm", "face_vomiting", "face_with_cowboy_hat", "face_with_diagonal_mouth", "face_with_hand_over_mouth", "face_with_head_bandage",
  "face_with_monocle", "face_with_open_eyes_and_hand_over_mouth", "face_with_peeking_eye", "face_with_raised_eyebrow", "face_with_rolling_eyes", "face_with_spiral_eyes", "face_with_symbols_on_mouth", "face_with_thermometer",
  "facepunch", "factory", "factory_worker", "fairy", "falafel", "fallen_leaf", "family", "farmer",
  "fast_forward", "fax", "fearful", "feather", "feet", "female-artist", "female-astronaut", "female-construction-worker",
  "female-cook", "female-detective", "female-doctor", "female-factory-worker", "female-farmer", "female-firefighter", "female-guard", "female-judge",
  "female-mechanic", "female-office-worker", "female-pilot", "female-police-officer", "female-scientist", "female-singer", "female-student", "female-teacher",
  "female-technologist", "female_elf", "female_fairy", "female_genie", "female_mage", "female_sign", "female_superhero", "female_supervillain",
  "female_vampire", "female_zombie", "fencer", "ferris_wheel", "ferry", "field_hockey_stick_and_ball", "file_cabinet", "file_folder",
  "film_frames", "film_projector", "fire", "fire_engine", "fire_extinguisher", "firecracker", "firefighter", "fireworks",
  "first_place_medal", "first_quarter_moon", "first_quarter_moon_with_face", "fish", "fish_cake", "fishing_pole_and_fish", "fist", "five",
  "flags", "flamingo", "flashlight", "flatbread", "fleur_de_lis", "floppy_disk", "flower_playing_cards", "flushed",
  "fly", "flying_disc", "flying_saucer", "fog", "foggy", "fondue", "foot", "football",
  "footprints", "fork_and_knife", "fortune_cookie", "fountain", "four", "four_leaf_clover", "fox_face", "fr",
  "frame_with_picture", "free", "fried_egg", "fried_shrimp", "fries", "frog", "frowning", "fuelpump",
  "full_moon", "full_moon_with_face", "funeral_urn", "game_die", "garlic", "gb", "gear", "gem",
  "gemini", "genie", "ghost", "gift", "gift_heart", "giraffe_face", "girl", "glass_of_milk",
  "globe_with_meridians", "gloves", "goal_net", "goat", "goggles", "golf", "golfer", "gorilla",
  "grapes", "green_apple", "green_book", "green_heart", "green_salad", "grey_exclamation", "grey_question", "grimacing",
  "grin", "grinning", "guardsman", "guide_dog", "guitar", "gun", "haircut", "hamburger",
  "hammer", "hammer_and_pick", "hammer_and_wrench", "hamsa", "hamster", "hand", "hand_with_index_finger_and_thumb_crossed", "handbag",
  "handball", "handshake", "hankey", "hash", "hatched_chick", "hatching_chick", "headphones", "headstone",
  "health_worker", "hear_no_evil", "heart", "heart_decoration", "heart_eyes", "heart_eyes_cat", "heart_hands", "heart_on_fire",
  "heartbeat", "heartpulse", "hearts", "heavy_check_mark", "heavy_division_sign", "heavy_dollar_sign", "heavy_equals_sign", "heavy_heart_exclamation_mark_ornament",
  "heavy_minus_sign", "heavy_multiplication_x", "heavy_plus_sign", "hedgehog", "helicopter", "helmet_with_white_cross", "herb", "hibiscus",
  "high_brightness", "high_heel", "hiking_boot", "hindu_temple", "hippopotamus", "hocho", "hole", "honey_pot",
  "hook", "horse", "horse_racing", "hospital", "hot_face", "hot_pepper", "hotdog", "hotel",
  "hotsprings", "hourglass", "hourglass_flowing_sand", "house", "house_buildings", "house_with_garden", "hugging_face", "hushed",
  "hut", "i_love_you_hand_sign", "ice_cream", "ice_cube", "ice_hockey_stick_and_puck", "ice_skate", "icecream", "id",
  "identification_card", "ideograph_advantage", "imp", "inbox_tray", "incoming_envelope", "index_pointing_at_the_viewer", "infinity", "information_desk_person",
  "information_source", "innocent", "interrobang", "iphone", "it", "izakaya_lantern", "jack_o_lantern", "japan",
  "japanese_castle", "japanese_goblin", "japanese_ogre", "jar", "jeans", "jigsaw", "joy", "joy_cat",
  "joystick", "jp", "judge", "juggling", "kaaba", "kangaroo", "key", "keyboard",
  "keycap_star", "keycap_ten", "kimono", "kiss", "kissing", "kissing_cat", "kissing_closed_eyes", "kissing_heart",
  "kissing_smiling_eyes", "kite", "kiwifruit", "kneeling_person", "knife_fork_plate", "knot", "koala", "koko",
  "kr", "lab_coat", "label", "lacrosse", "ladder", "ladybug", "large_blue_circle", "large_blue_diamond",
  "large_blue_square", "large_brown_circle", "large_brown_square", "large_green_circle", "large_green_square", "large_orange_circle", "large_orange_diamond", "large_orange_square",
  "large_purple_circle", "large_purple_square", "large_red_square", "large_yellow_circle", "large_yellow_square", "last_quarter_moon", "last_quarter_moon_with_face", "latin_cross",
  "laughing", "leafy_green", "leaves", "ledger", "left-facing_fist", "left_luggage", "left_right_arrow", "left_speech_bubble",
  "leftwards_arrow_with_hook", "leftwards_hand", "leg", "lemon", "leo", "leopard", "level_slider", "libra",
  "light_rail", "lightning", "link", "linked_paperclips", "lion_face", "lips", "lipstick", "lizard",
  "llama", "lobster", "lock", "lock_with_ink_pen", "lollipop", "long_drum", "loop", "lotion_bottle",
  "lotus", "loud_sound", "loudspeaker", "love_hotel", "love_letter", "low_battery", "low_brightness", "lower_left_ballpoint_pen",
  "lower_left_crayon", "lower_left_fountain_pen", "lower_left_paintbrush", "luggage", "lungs", "lying_face", "m", "mag",
  "mag_right", "mage", "magic_wand", "magnet", "mahjong", "mailbox", "mailbox_closed", "mailbox_with_mail",
  "mailbox_with_no_mail", "male-artist", "male-astronaut", "male-construction-worker", "male-cook", "male-detective", "male-doctor", "male-factory-worker",
  "male-farmer", "male-firefighter", "male-guard", "male-judge", "male-mechanic", "male-office-worker", "male-pilot", "male-police-officer",
  "male-scientist", "male-singer", "male-student", "male-teacher", "male-technologist", "male_elf", "male_fairy", "male_genie",
  "male_mage", "male_sign", "male_superhero", "male_supervillain", "male_vampire", "male_zombie", "mammoth", "man",
  "man-biking", "man-bouncing-ball", "man-bowing", "man-boy", "man-boy-boy", "man-cartwheeling", "man-facepalming", "man-frowning",
  "man-gesturing-no", "man-gesturing-ok", "man-getting-haircut", "man-getting-massage", "man-girl", "man-girl-boy", "man-girl-girl", "man-golfing",
  "man-heart-man", "man-juggling", "man-kiss-man", "man-lifting-weights", "man-man-boy", "man-man-boy-boy", "man-man-girl", "man-man-girl-boy",
  "man-man-girl-girl", "man-mountain-biking", "man-playing-handball", "man-playing-water-polo", "man-pouting", "man-raising-hand", "man-rowing-boat", "man-running",
  "man-shrugging", "man-surfing", "man-swimming", "man-tipping-hand", "man-walking", "man-wearing-turban", "man-woman-boy", "man-woman-boy-boy",
  "man-woman-girl", "man-woman-girl-boy", "man-woman-girl-girl", "man-wrestling", "man_and_woman_holding_hands", "man_climbing", "man_dancing", "man_feeding_baby",
  "man_in_business_suit_levitating", "man_in_lotus_position", "man_in_manual_wheelchair", "man_in_motorized_wheelchair", "man_in_steamy_room", "man_in_tuxedo", "man_kneeling", "man_standing",
  "man_with_beard", "man_with_gua_pi_mao", "man_with_probing_cane", "man_with_turban", "man_with_veil", "mango", "mans_shoe", "mantelpiece_clock",
  "manual_wheelchair", "maple_leaf", "martial_arts_uniform", "mask", "massage", "mate_drink", "meat_on_bone", "mechanic",
  "mechanical_arm", "mechanical_leg", "medal", "medical_symbol", "mega", "melon", "melting_face", "memo",
  "men-with-bunny-ears-partying", "mending_heart", "menorah_with_nine_branches", "mens", "mermaid", "merman", "merperson", "metro",
  "microbe", "microphone", "microscope", "middle_finger", "military_helmet", "milky_way", "minibus", "minidisc",
  "mirror", "mirror_ball", "mobile_phone_off", "money_mouth_face", "money_with_wings", "moneybag", "monkey", "monkey_face",
  "monorail", "moon", "moon_cake", "mortar_board", "mosque", "mosquito", "mostly_sunny", "motor_boat",
  "motor_scooter", "motorized_wheelchair", "motorway", "mount_fuji", "mountain", "mountain_bicyclist", "mountain_cableway", "mountain_railway",
  "mouse", "mouse2", "mouse_trap", "movie_camera", "moyai", "mrs_claus", "muscle", "mushroom",
  "musical_keyboard", "musical_note", "musical_score", "mute", "mx_claus", "nail_care", "name_badge", "national_park",
  "nauseated_face", "nazar_amulet", "necktie", "negative_squared_cross_mark", "nerd_face", "nest_with_eggs", "nesting_dolls", "neutral_face",
  "new", "new_moon", "new_moon_with_face", "newspaper", "ng", "night_with_stars", "nine", "ninja",
  "no_bell", "no_bicycles", "no_entry", "no_entry_sign", "no_good", "no_mobile_phones", "no_mouth", "no_pedestrians",
  "no_smoking", "non-potable_water", "nose", "notebook", "notebook_with_decorative_cover", "notes", "nut_and_bolt", "o",
  "o2", "ocean", "octagonal_sign", "octopus", "oden", "office", "office_worker", "oil_drum",
  "ok", "ok_hand", "ok_woman", "old_key", "older_adult", "older_man", "older_woman", "olive",
  "om_symbol", "on", "oncoming_automobile", "oncoming_bus", "oncoming_police_car", "oncoming_taxi", "one", "one-piece_swimsuit",
  "onion", "open_file_folder", "open_hands", "open_mouth", "ophiuchus", "orange_book", "orange_heart", "orangutan",
  "orthodox_cross", "otter", "outbox_tray", "owl", "ox", "oyster", "package", "page_facing_up",
  "page_with_curl", "pager", "palm_down_hand", "palm_tree", "palm_up_hand", "palms_up_together", "pancakes", "panda_face",
  "paperclip", "parachute", "parking", "parrot", "part_alternation_mark", "partly_sunny", "partly_sunny_rain", "partying_face",
  "passenger_ship", "passport_control", "peace_symbol", "peach", "peacock", "peanuts", "pear", "pencil2",
  "penguin", "pensive", "people_holding_hands", "people_hugging", "performing_arts", "persevere", "person_climbing", "person_doing_cartwheel",
  "person_feeding_baby", "person_frowning", "person_in_lotus_position", "person_in_manual_wheelchair", "person_in_motorized_wheelchair", "person_in_steamy_room", "person_in_tuxedo", "person_with_ball",
  "person_with_blond_hair", "person_with_crown", "person_with_headscarf", "person_with_pouting_face", "person_with_probing_cane", "petri_dish", "phone", "pick",
  "pickup_truck", "pie", "pig", "pig2", "pig_nose", "pill", "pilot", "pinata",
  "pinched_fingers", "pinching_hand", "pineapple", "pirate_flag", "pisces", "pizza", "placard", "place_of_worship",
  "playground_slide", "pleading_face", "plunger", "point_down", "point_left", "point_right", "point_up", "point_up_2",
  "polar_bear", "police_car", "poodle", "popcorn", "post_office", "postal_horn", "postbox", "potable_water",
  "potato", "potted_plant", "pouch", "poultry_leg", "pound", "pouring_liquid", "pouting_cat", "pray",
  "prayer_beads", "pregnant_man", "pregnant_person", "pregnant_woman", "pretzel", "prince", "princess", "printer",
  "probing_cane", "purple_heart", "purse", "pushpin", "put_litter_in_its_place", "question", "rabbit", "rabbit2",
  "raccoon", "racehorse", "racing_car", "racing_motorcycle", "radio", "radio_button", "radioactive_sign", "rage",
  "railway_car", "railway_track", "rain_cloud", "rainbow", "rainbow-flag", "raised_back_of_hand", "raised_hand_with_fingers_splayed", "raised_hands",
  "raising_hand", "ram", "ramen", "rat", "razor", "receipt", "recycle", "red_circle",
  "red_envelope", "red_haired_man", "red_haired_person", "red_haired_woman", "registered", "relaxed", "relieved", "reminder_ribbon",
  "repeat", "repeat_one", "restroom", "revolving_hearts", "rewind", "rhinoceros", "ribbon", "rice",
  "rice_ball", "rice_cracker", "rice_scene", "right-facing_fist", "right_anger_bubble", "rightwards_hand", "ring", "ring_buoy",
  "ringed_planet", "robot_face", "rock", "rocket", "roll_of_paper", "rolled_up_newspaper", "roller_coaster", "roller_skate",
  "rolling_on_the_floor_laughing", "rooster", "rose", "rosette", "rotating_light", "round_pushpin", "rowboat", "ru",
  "rugby_football", "runner", "running_shirt_with_sash", "sa", "safety_pin", "safety_vest", "sagittarius", "sake",
  "salt", "saluting_face", "sandal", "sandwich", "santa", "sari", "satellite", "satellite_antenna",
  "sauropod", "saxophone", "scales", "scarf", "school", "school_satchel", "scientist", "scissors",
  "scooter", "scorpion", "scorpius", "scream", "scream_cat", "screwdriver", "scroll", "seal",
  "seat", "second_place_medal", "secret", "see_no_evil", "seedling", "selfie", "service_dog", "seven",
  "sewing_needle", "shallow_pan_of_food", "shamrock", "shark", "shaved_ice", "sheep", "shell", "shield",
  "shinto_shrine", "ship", "shirt", "shopping_bags", "shopping_trolley", "shorts", "shower", "shrimp",
  "shrug", "shushing_face", "signal_strength", "singer", "six", "six_pointed_star", "skateboard", "ski",
  "skier", "skin-tone-2", "skin-tone-3", "skin-tone-4", "skin-tone-5", "skin-tone-6", "skull", "skull_and_crossbones",
  "skunk", "sled", "sleeping", "sleeping_accommodation", "sleepy", "sleuth_or_spy", "slightly_frowning_face", "slightly_smiling_face",
  "slot_machine", "sloth", "small_airplane", "small_blue_diamond", "small_orange_diamond", "small_red_triangle", "small_red_triangle_down", "smile",
  "smile_cat", "smiley", "smiley_cat", "smiling_face_with_3_hearts", "smiling_face_with_tear", "smiling_imp", "smirk", "smirk_cat",
  "smoking", "snail", "snake", "sneezing_face", "snow_capped_mountain", "snow_cloud", "snowboarder", "snowflake",
  "snowman", "snowman_without_snow", "soap", "sob", "soccer", "socks", "softball", "soon",
  "sos", "sound", "space_invader", "spades", "spaghetti", "sparkle", "sparkler", "sparkles",
  "sparkling_heart", "speak_no_evil", "speaker", "speaking_head_in_silhouette", "speech_balloon", "speedboat", "spider", "spider_web",
  "spiral_calendar_pad", "spiral_note_pad", "spock-hand", "sponge", "spoon", "sports_medal", "squid", "stadium",
  "standing_person", "star", "star-struck", "star2", "star_and_crescent", "star_of_david", "stars", "station",
  "statue_of_liberty", "steam_locomotive", "stethoscope", "stew", "stopwatch", "straight_ruler", "strawberry", "stuck_out_tongue",
  "stuck_out_tongue_closed_eyes", "stuck_out_tongue_winking_eye", "student", "studio_microphone", "stuffed_flatbread", "sun_with_face", "sunflower", "sunglasses",
  "sunny", "sunrise", "sunrise_over_mountains", "superhero", "supervillain", "surfer", "sushi", "suspension_railway",
  "swan", "sweat", "sweat_drops", "sweat_smile", "sweet_potato", "swimmer", "symbols", "synagogue",
  "syringe", "t-rex", "table_tennis_paddle_and_ball", "taco", "tada", "takeout_box", "tamale", "tanabata_tree",
  "tangerine", "taurus", "taxi", "tea", "teacher", "teapot", "technologist", "teddy_bear",
  "telephone_receiver", "telescope", "tennis", "tent", "test_tube", "the_horns", "thermometer", "thinking_face",
  "third_place_medal", "thong_sandal", "thought_balloon", "thread", "three", "three_button_mouse", "thunder_cloud_and_rain", "ticket",
  "tiger", "tiger2", "timer_clock", "tired_face", "tm", "toilet", "tokyo_tower", "tomato",
  "tongue", "toolbox", "tooth", "toothbrush", "top", "tophat", "tornado", "trackball",
  "tractor", "traffic_light", "train", "train2", "tram", "transgender_flag", "transgender_symbol", "triangular_flag_on_post",
  "triangular_ruler", "trident", "triumph", "troll", "trolleybus", "trophy", "tropical_drink", "tropical_fish",
  "truck", "trumpet", "tulip", "tumbler_glass", "turkey", "turtle", "tv", "twisted_rightwards_arrows",
  "two", "two_hearts", "two_men_holding_hands", "two_women_holding_hands", "umbrella", "umbrella_on_ground", "umbrella_with_rain_drops", "unamused",
  "underage", "unicorn_face", "unlock", "up", "upside_down_face", "us", "v", "vampire",
  "vertical_traffic_light", "vhs", "vibration_mode", "video_camera", "video_game", "violin", "virgo", "volcano",
  "volleyball", "vs", "waffle", "walking", "waning_crescent_moon", "waning_gibbous_moon", "warning", "wastebasket",
  "watch", "water_buffalo", "water_polo", "watermelon", "wave", "waving_black_flag", "waving_white_flag", "wavy_dash",
  "waxing_crescent_moon", "wc", "weary", "wedding", "weight_lifter", "whale", "whale2", "wheel",
  "wheel_of_dharma", "wheelchair", "white_check_mark", "white_circle", "white_flower", "white_frowning_face", "white_haired_man", "white_haired_person",
  "white_haired_woman", "white_heart", "white_large_square", "white_medium_small_square", "white_medium_square", "white_small_square", "white_square_button", "wilted_flower",
  "wind_blowing_face", "wind_chime", "window", "wine_glass", "wink", "wolf", "woman", "woman-biking",
  "woman-bouncing-ball", "woman-bowing", "woman-boy", "woman-boy-boy", "woman-cartwheeling", "woman-facepalming", "woman-frowning", "woman-gesturing-no",
  "woman-gesturing-ok", "woman-getting-haircut", "woman-getting-massage", "woman-girl", "woman-girl-boy", "woman-girl-girl", "woman-golfing", "woman-heart-man",
  "woman-heart-woman", "woman-juggling", "woman-kiss-man", "woman-kiss-woman", "woman-lifting-weights", "woman-mountain-biking", "woman-playing-handball", "woman-playing-water-polo",
  "woman-pouting", "woman-raising-hand", "woman-rowing-boat", "woman-running", "woman-shrugging", "woman-surfing", "woman-swimming", "woman-tipping-hand",
  "woman-walking", "woman-wearing-turban", "woman-woman-boy", "woman-woman-boy-boy", "woman-woman-girl", "woman-woman-girl-boy", "woman-woman-girl-girl", "woman-wrestling",
  "woman_climbing", "woman_feeding_baby", "woman_in_lotus_position", "woman_in_manual_wheelchair", "woman_in_motorized_wheelchair", "woman_in_steamy_room", "woman_in_tuxedo", "woman_kneeling",
  "woman_standing", "woman_with_beard", "woman_with_probing_cane", "woman_with_veil", "womans_clothes", "womans_flat_shoe", "womans_hat", "women-with-bunny-ears-partying",
  "womens", "wood", "woozy_face", "world_map", "worm", "worried", "wrench", "wrestlers",
  "writing_hand", "x", "x-ray", "yarn", "yawning_face", "yellow_heart", "yen", "yin_yang",
  "yo-yo", "yum", "zany_face", "zap", "zebra_face", "zero", "zipper_mouth_face", "zombie",
  "zzz"
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

def closest_emoji(emoji):
    ''' :thinking: '''
    emoji = emoji.strip(':')
    best_score = 0
    best_emoji = 'heart'
    for candidate in all_emoji:
        # try harder for _face emoji
        for attempt in (emoji, f"{emoji}_face"):
            score = ratio(candidate, attempt)
            if score > best_score:
                best_score = score
                best_emoji = candidate
    return f":{best_emoji}:"

def get_spectrum(score):
    ''' Translate a score from -1 to 1 into an emoji on the spectrum '''
    return spectrum_emoji[int(((score + 1) / 2) * (len(spectrum_emoji) - 1))]

def get_degree(score):
    ''' Turn a 0.0-1.0 score into a degree '''
    return degrees[int(score * (len(degrees) - 1))]

def get_profanity_score(prompt):
    ''' Profanity analysis with slkearn. Returns a float, -1.0 to 0 '''
    return -profanity_prob([prompt])[0]

def get_flair_score(prompt):
    ''' Run the flair sentiment prediction model. Returns a float, -1.0 to 1.0 '''
    global flair_sentiment
    if not flair_sentiment:
         flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
    sent = flair.data.Sentence(prompt)
    flair_sentiment.predict(sent)

    if sent.labels[0].value == 'NEGATIVE':
        return -sent.labels[0].score

    return sent.labels[0].score

class Sentiment(object):
    def __init__(self, engine="flair", model=None):
        self.engine = engine
        # `model` doesn't do anything for Flair right now because we're not
        # loading the Flair model here
        self.model = model

        if self.engine == "spacy":
            self.nlp = spacy.load(self.model or "en_core_web_lg")
            self.nlp.add_pipe('spacytextblob')

    def get_sentiment_score(self, prompt):
        if self.engine == "spacy":
            doc = self.nlp(prompt)
            return doc._.blob.polarity
        else:
            return get_flair_score(prompt)

    def get_profanity_score(self, prompt):
        return get_profanity_score(prompt)