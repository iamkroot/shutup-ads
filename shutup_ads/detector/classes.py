"""AudioSet class definitions and indices relevant to sports and commercials."""

from __future__ import annotations

# Basketball & Live Sports Event Indices
BASKETBALL_BOUNCE = 459   # "Basketball bounce"
SQUEAK = 355              # "Squeak" (sneaker squeaks on hardwood court)
WHISTLE = 396             # "Whistle" (referee whistle)
BUZZER = 392              # "Buzzer" (shot clock / game buzzer)
CROWD = 64                # "Crowd"
CHEERING = 61             # "Cheering"
APPLAUSE = 62             # "Applause"
SCREAMING = 11            # "Screaming"
SHOUT = 6                 # "Shout"
YELL = 9                  # "Yell"

BASKETBALL_SPORTS_CLASSES = [
    BASKETBALL_BOUNCE,
    SQUEAK,
    WHISTLE,
    BUZZER,
    CROWD,
    CHEERING,
    APPLAUSE,
]

# Commercial Break & Advertising Indices
MUSIC = 132               # "Music"
BACKGROUND_MUSIC = 262    # "Background music"
POP_MUSIC = 211           # "Pop music"
JINGLE_MUSIC = 264        # "Jingle (music)"
THEME_MUSIC = 263         # "Theme music"
SOUNDTRACK_MUSIC = 265    # "Soundtrack music"
HIP_HOP_MUSIC = 212       # "Hip hop music"
ROCK_MUSIC = 214          # "Rock music"
RADIO = 519               # "Radio"
TELEVISION = 518          # "Television"

COMMERCIAL_MUSIC_CLASSES = [
    MUSIC,
    BACKGROUND_MUSIC,
    POP_MUSIC,
    JINGLE_MUSIC,
    THEME_MUSIC,
    SOUNDTRACK_MUSIC,
    HIP_HOP_MUSIC,
    ROCK_MUSIC,
]

# Speech Indices
SPEECH = 0                # "Speech"
NARRATION = 3             # "Narration, monologue"
CONVERSATION = 2          # "Conversation"

SPEECH_CLASSES = [
    SPEECH,
    NARRATION,
    CONVERSATION,
]

# Silence & Transition Indicators
SILENCE = 494             # "Silence"
