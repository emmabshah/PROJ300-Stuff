"""
Freddy — Main Control Script
================================================
An animatronic parrot robot with:
  - Dual-camera face tracking (eyes & neck move)
  - Facial emotion detection with reactive behaviours
  - Wake-word ("Hey Freddy") triggered Parrot Mode
  - TTS with beak animation
  - Emotion-driven eyelid, wing, and LED behaviours
  - Scanning idle behaviour when no face is present
"""

import cv2
import warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype")
import mediapipe as mp
import board
import busio
import time
import random
import threading
import csv
import datetime
from collections import deque, Counter
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from adafruit_pca9685 import PCA9685
from picamera2 import Picamera2
import neopixel
import pyaudio
import pvporcupine
import numpy as np
import wave
import math
import re
import subprocess
import os
from better_profanity import profanity
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv("/home/student/freddy/.env")


# =============================================================================
#  SECTION 1: CONFIGURATION FLAG
# =============================================================================

# Set True to write per-frame tracking and detection data to CSV files
# in the logging directory. 
# When enabled, the script will prompt for a test condition label at
# startup
#LOGGING_ENABLED = True
LOGGING_ENABLED = False

LOG_DIR = "/home/student/freddy/logs"


# =============================================================================
#  SECTION 2: HARDWARE SETUP
# =============================================================================

# ── MediaPipe face detection ─────────────────────────────────────────────────
# model_selection=0 selects the short-range model (faces within ~2 m),
# which suits Freddy's typical interaction distance for audio detection.
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)

# ── PCA9685 servo driver ─────────────────────────────────────────────────────
# Details from Arduino servo starter code
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.reference_clock_speed = 27_000_000
pca.frequency = 50


# =============================================================================
#  SECTION 3: SERVO CHANNEL DEFINITIONS & LIMITS
# =============================================================================
# Each servo has a channel on the PCA9685 and calibrated tick values for its
# mechanical limits.  Tick values were found experimentally per servo.

# ── Left eye ─────────────────────────────────────────────────────────────────
L_TILT_CH   = 5
L_TILT_UP   = 240
L_TILT_CEN  = 303
L_TILT_DOWN = 366

L_PAN_CH    = 6
L_PAN_LEFT  = 228
L_PAN_CEN   = 378   # intentionally matches RIGHT — mechanical range is one-sided
L_PAN_RIGHT = 378

# ── Right eye ────────────────────────────────────────────────────────────────
R_TILT_CH   = 9
R_TILT_UP   = 380
R_TILT_CEN  = 310
R_TILT_DOWN = 245

R_PAN_CH    = 10
R_PAN_LEFT  = 226
R_PAN_CEN   = 226   # intentionally matches LEFT — mechanical range is one-sided
R_PAN_RIGHT = 369

# ── Eyelids ──────────────────────────────────────────────────────────────────
L_EYELID_CH     = 4
L_EYELID_OPEN   = 320
L_EYELID_CLOSED = 225

R_EYELID_CH     = 8
R_EYELID_OPEN   = 283
R_EYELID_CLOSED = 380

# ── Neck ─────────────────────────────────────────────────────────────────────
NECK_CH    = 14
NECK_LEFT  = 180
NECK_CEN   = 230
NECK_RIGHT = 280

# ── Beak ─────────────────────────────────────────────────────────────────────
BEAK_CH     = 11
BEAK_CLOSED = 250
BEAK_OPEN   = 180

# ── Wings ────────────────────────────────────────────────────────────────────
L_WING_CH     = 0
L_WING_CLOSED = 280
L_WING_OPEN   = 183

R_WING_CH     = 15
R_WING_CLOSED = 311
R_WING_OPEN   = 418


# =============================================================================
#  SECTION 4: LED BELLY SETUP
# =============================================================================

PIXEL_PIN  = board.D18
NUM_PIXELS = 16
BRIGHTNESS = 0.2

pixels = neopixel.NeoPixel(
    PIXEL_PIN,
    NUM_PIXELS,
    brightness=BRIGHTNESS,
    auto_write=False,
    pixel_order=neopixel.GRB,
)

# LED colour per emotion (R, G, B) — chosen to be visually distinct and
# intuitively associated with each emotion.
EMOTION_LED_COLOURS = {
    "happy":    (255, 255,   0),  # yellow
    "sad":      (  0, 100, 255),  # blue
    "angry":    (255,   0,   0),  # red
    "surprise": (255,  20, 147),  # pink
    "fear":     ( 20,   0,  40),  # purple
    "disgust":  (  0, 180,   0),  # green
    "neutral":  (130,  35,   0),  # orange
    "unknown":  (  0,   0,   0),  # black
}

# Debug overlay colours (BGR for OpenCV)
EMOTION_COLOURS = {
    "happy":    (  0, 255, 255),  # yellow
    "sad":      (255, 100,   0),  # blue
    "angry":    (  0,   0, 255),  # red
    "surprise": (147,  20, 255),  # pink
    "fear":     (180,   0, 128),  # purple
    "disgust":  (  0, 180,   0),  # green
    "neutral":  (  0,  90, 200),  # orange
    "unknown":  (  0,   0,   0),  # black
}


# =============================================================================
#  SECTION 5: AUDIO & VOICE SETTINGS
# =============================================================================

# ── Porcupine wake word ──────────────────────────────────────────────────────
# Access key loaded from environment variable to avoid sharing sensitive information.
ACCESS_KEY   = os.getenv("PORCUPINE_ACCESS_KEY", "")
KEYWORD_PATH = "/home/student/freddy/hey_freddy.ppn"

# ── Recording settings ───────────────────────────────────────────────────────
FORMAT         = pyaudio.paInt16
CHANNELS       = 1
# How long Freddy listens after wake-word detection
LISTEN_SECONDS = 5.0            
OUTPUT_WAV     = "/home/student/freddy/command.wav"

# ── Whisper STT ───────────────────────────────────────────────────
# Using whisper.cpp for fast on-device inference.
# The --translate flag makes Whisper output English regardless of input language.
# The 'base' model was chosen as the best trade-off between accuracy and speed.
WHISPER_CLI     = "/home/student/freddy/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL   = "/home/student/freddy/whisper.cpp/models/ggml-base.bin"
WHISPER_THREADS = "4"

# ── Piper TTS────────────────────────────────────────────────────
# Custom parrot voice trained with TextyMcSpeechy on Chatterbox-generated
# training data, using an ElevenLabs voice.
PIPER_EXE          = "/home/student/freddy/piper/piper"
PIPER_VOICE_DIR    = "/home/student/new_tts_voices/parrot_4734"
PIPER_MODEL        = os.path.join(PIPER_VOICE_DIR, "en_GB-parrot_4734-medium.onnx")
PIPER_LENGTH_SCALE = "0.85"    # Slightly faster than default speech rate
PIPER_NOISE_SCALE  = "0.2"     # Low variability for consistent character voice
PIPER_NOISE_W      = "0.5"
TTS_WAV            = "/tmp/parrot_tts.wav"

# ── Squawk audio files ──────────────────────────────────────────────────────
# Pre-recorded squawk WAVs generated with ElevenLabs, one set per emotion
# and a generic set for after Parrot Mode speech.
EMOTION_SQUAWK_DIR = "/home/student/freddy/Emotion Detection Squawks"
AFTER_SPEECH_DIR   = "/home/student/freddy/After Speech Squawks"
EMOTION_SQUAWK_COUNTS = {
    "happy": 6, "angry": 6, "fear": 6,
    "sad": 6, "surprise": 6, "neutral": 6, "disgust": 6,
}
AFTER_SPEECH_COUNT = 10


# =============================================================================
#  SECTION 6: TRACKING PARAMETERS
# =============================================================================

# Per-camera axis sign corrections.
# Flips the error direction to account for camera mounting position.
CAM_EX_SIGN = {"L": -1, "R": -1}
CAM_EY_SIGN = {"L":  1, "R":  1}

# A proportional controller is used rather than full PID because the servos
# have no positional feedback.
DEADZONE_TILT = 30        # Pixels of error ignored to prevent micro-jitter
DEADZONE_PAN  = 30        # Pixels of error ignored to prevent micro-jitter
Kp_EYE        = 0.12      # Proportional gain: higher = more responsive but shakier
MIN_STEP_EYE  = 0.002     # Minimum movement per frame (filters out noise)
MAX_STEP_EYE  = 0.08      # Maximum movement per frame (caps speed to avoid overshooting)

# Neck tracking gains: slower than eyes for natural head-follows-eyes look.
NECK_DEADZONE = 20
Kp_NECK       = 0.05
MIN_STEP_NECK = 1
MAX_STEP_NECK = 3

# How fast the inactive eye (not tracking) drifts back to centre.
# Creates a natural convergence effect where both eyes settle forward.
INACTIVE_RETURN_SPEED = 0.04

# How long to hold the active camera target before re-evaluating.
# Prevents rapid switching between left/right camera when a face is
# visible in both and keeps tracking smooth.
TARGET_HOLD_TIME = 1.0


# =============================================================================
#  SECTION 7: EMOTION PARAMETERS
# =============================================================================
#
# Each emotion has parameters controlling three animation subsystems:
#
#   Wings:   w_target   : how far wings open (0=closed, 1=full)
#            w_raise    : speed of raising per frame
#            w_lower    : speed of lowering per frame
#            w_hold     : (min, max) seconds held at top
#            w_wait     : (min, max) seconds between wing cycles
#            w_flutters : number of up-down cycles per movement
#
#   Eyelids: e_rest     : resting openness (0=wide open, 1=closed)
#            e_speed    : blink speed multiplier
#            e_hold     : seconds eyes stay shut during blink
#
#   Tilt:    tilt_limit : max vertical tracking range (0-1),
#                         e.g. angry=0 means no vertical tracking (glaring)

EMOTION_PARAMS = {
    "happy":    dict(w_target=0.85, w_raise=0.06, w_lower=0.06, w_hold=(0.05, 0.10), w_wait=(0.8,  2.0),  w_flutters=4, e_rest=0.25, e_speed=2.0,  e_hold=0.05, tilt_limit=0.8),
    "angry":    dict(w_target=1.0,  w_raise=0.03, w_lower=0.02, w_hold=(1.5,  2.5),  w_wait=(3.0,  6.0),  w_flutters=2, e_rest=0.8,  e_speed=2.0,  e_hold=0.05, tilt_limit=0.0),
    "surprise": dict(w_target=1.0,  w_raise=0.10, w_lower=0.03, w_hold=(0.5,  1.0),  w_wait=(3.0,  5.0),  w_flutters=1, e_rest=0.0,  e_speed=0.5,  e_hold=0.05, tilt_limit=1.0),
    "sad":      dict(w_target=0.5,  w_raise=0.02, w_lower=0.02, w_hold=(1.0,  2.0),  w_wait=(4.0,  8.0),  w_flutters=1, e_rest=0.5,  e_speed=0.5,  e_hold=0.20, tilt_limit=0.4),
    "fear":     dict(w_target=1.0,  w_raise=0.10, w_lower=0.10, w_hold=(0.02, 0.05), w_wait=(0.5,  1.5),  w_flutters=5, e_rest=0.0,  e_speed=2.0,  e_hold=0.05, tilt_limit=1.0),
    "disgust":  dict(w_target=1.0,  w_raise=0.03, w_lower=0.03, w_hold=(0.5,  1.0),  w_wait=(3.0,  6.0),  w_flutters=2, e_rest=0.65, e_speed=1.0,  e_hold=0.05, tilt_limit=0.2),
    "neutral":  dict(w_target=0.7,  w_raise=0.02, w_lower=0.02, w_hold=(0.3,  0.8),  w_wait=(4.0,  9.0),  w_flutters=2, e_rest=0.35, e_speed=1.75, e_hold=0.05, tilt_limit=0.6),
    "unknown":  dict(w_target=0.3,  w_raise=0.02, w_lower=0.02, w_hold=(0.0,  0.0),  w_wait=(5.0, 10.0),  w_flutters=1, e_rest=0.35, e_speed=1.75, e_hold=0.05, tilt_limit=0.6),
}

# ── Emotion detection timing ─────────────────────────────────────────────────
EMOTION_INTERVAL    = 0.5   # seconds between inference runs (keeps CPU load manageable)
EMOTION_HISTORY     = 2     # recent predictions kept for majority voting
EMOTION_STABLE_TIME = 1.0   # seconds an emotion must persist to become "stable" which prevents reacting to noisy single-frame predictions
EMOTION_MIN_CONF    = 50.0  # minimum confidence (%) to accept an emotion prediction: low-confidence guesses are discarded before voting
MIN_FACE_PX         = 60    # minimum face crop size in pixels (smaller crops are too noisy for reliable emotion inference)
FACE_PAD            = 0.20  # padding around face bounding box: gives the model surrounding context which improves prediction quality


# =============================================================================
#  SECTION 8: PHRASE BANKS
# =============================================================================

DIDNT_HEAR_PHRASES = [
    "Oi, what did you say?",
    "Sorry, Freddy didn't hear that!",
    "Blimey, I didn't catch that!",
    "Shiver me feathers, say that again?",
    "My ears must be full of seawater! Can you repeat that?",
]

EMOTION_PHRASES = {
    "happy": [
        "Freddy is so happy to meet a new friend!",
        "Shiver me feathers, you've made my day!",
        "Blimey, that smile could light up the seven seas!",
        "Freddy loves a happy visitor, yo ho ho!",
        "You're as bright as a tropical sunrise!",
        "What a jolly good day you're having!",
        "Blow me down, you've got me feeling fantastic!",
        "That smile is worth more than all the gold in Davy Jones' locker!",
        "Freddy wants to feel as happy as you!",
        "You're making me flap my wings with joy!",
        "Ahoy, happiness is in the air today!",
        "By the compass rose, what a wonderful day this is!",
        "Your smile has me singing like a sea angel!",
        "You've got me grinning from beak to tail feather!",
        "Shiver me feathers, this is the best day ever!",
        "Freddy could soar through the sky with happiness!",
        "Your happiness is contagious, I caught it!",
        "Blimey, you look like you found buried treasure!",
        "Hoist the jolly roger, it's a brilliant day!",
        "My feathers are ruffled with delight!",
        "Yo ho ho, spread that joy around!",
        "You're sunnier than the Caribbean!",
        "Freddy loves your smile!",
        "That happiness is more precious than gold!",
        "By Blackbeard's beard, you're glowing today!",
        "My tail feathers are wagging with glee!",
        "All hands on deck, because your joy calls for a celebration!",
        "You've got more sparkle than the ocean on a sunny day!",
        "Freddy declares this the happiest ship on the seven seas!",
        "You look like you're on top of the world today, and I love it!",
    ],
    "angry": [
        "Blimey, someone woke up on the wrong side of the nest!",
        "Shiver me feathers, that's quite a scowl you've got there!",
        "Freddy doesn't like this energy one bit!",
        "Walk the plank if you must!",
        "By Davy Jones, you look angrier than a sea monster!",
        "Easy there sailor, no need to duel anyone today!",
        "I've seen storms calmer than your face right now!",
        "You seem to have a temper brewing!",
        "You look angrier than a pirate who lost his treasure!",
        "Now you've ruffled my feathers!",
        "Even Blackbeard was never this upset!",
        "That scowl could sink a ship!",
        "I hope that anger isn't aimed at me!",
        "Blow me down, what's got your feathers in a twist?",
        "A calm sea never made a skilled sailor, but neither did anger!",
        "Yo ho ho, shall we settle this squabble with a duel?",
        "Freddy's seen friendlier sharks!",
        "Even pirates know when to lower their swords!",
        "By the compass, that anger could navigate straight to trouble!",
        "Freddy thinks you need a cracker and some sea air!",
        "That temper is stormier than the North Atlantic!",
        "Shiver me feathers, what's the matter with you?",
        "Even Davy Jones lightens up sometimes!",
        "Freddy suggests counting to ten before firing the cannons!",
        "You look like you've swallowed a cannonball!",
        "By the seven seas, that's a face that could curdle rum!",
        "My wings are raised in anger!",
        "That glare is fiercer than the Bermuda Triangle!",
        "That face says it all!",
        "You look furious today matey, Freddy hopes it all blows over soon!",
    ],
    "sad": [
        "You look sad today, that makes me sad too!",
        "Shiver me feathers, those are cloudy eyes you've got!",
        "Freddy wishes you feel better soon!",
        "Even pirates cry sometimes, and that's alright!",
        "My feathers droop when you're sad!",
        "By Davy Jones, Freddy hopes the sun comes out for you soon!",
        "A sad face makes my beak quiver!",
        "Yo ho ho, let's find something to cheer you up!",
        "I would give you my favourite cracker if it would help!",
        "The seven seas have their rough patches too!",
        "Freddy sends you the warmest tropical squawk possible!",
        "Even the stormiest nights end in sunrise, friend!",
        "Blimey, Freddy wishes he could flap those clouds away!",
        "You look like you've lost your treasure map, poor soul!",
        "My heart is heavy seeing you like this!",
        "Shiver me feathers, I wish I could cheer you up!",
        "By the compass rose, sadness never lasts forever!",
        "Freddy would sail the seven seas to make you smile!",
        "Even the bravest pirates have blue days sometimes!",
        "Freddy thinks you deserve a hug and a cracker!",
        "Those sad eyes make me want to sing a gentle sea song!",
        "The tide always turns, friend, always!",
        "My wings are wrapped around you in spirit!",
        "By Blackbeard's beard, Freddy hopes tomorrow is brighter!",
        "Sadness visited me once, but I squawked it away!",
        "You look like the sea after a storm, still but recovering!",
        "Freddy believes in you more than all the gold in the Caribbean!",
        "Even lost ships find their way home eventually!",
        "Freddy will keep you company through these stormy waters!",
        "You're stronger than you know, friend, turn that frown upside down!",
    ],
    "surprise": [
        "Blow me down, you look like you've seen a ghost ship!",
        "Shiver me feathers, what in the seven seas happened to you!",
        "By Davy Jones' locker, those are wide eyes you've got!",
        "Freddy has never seen anyone look so surprised!",
        "Blimey, you look like you found a mermaid!",
        "Yo ho ho, what's got you so startled?",
        "My feathers nearly fell off seeing that expression!",
        "By the compass, something's really shocked you today!",
        "You look more surprised than a pirate finding empty treasure!",
        "Freddy didn't mean to startle you!",
        "Those eyes are wider than the eye of a hurricane!",
        "Shiver me feathers, you look absolutely astonished!",
        "By Blackbeard's beard, what a shocked face that is!",
        "Freddy wonders what could have caused such surprise!",
        "You look like a sea monster just winked at you!",
        "Blow me down, I'm surprised that you're so surprised!",
        "Even the bravest pirates look that shocked sometimes!",
        "That expression could stop a ship in full sail!",
        "Freddy hopes you got a good surprise!",
        "By the jolly roger, you look wonderfully shocked!",
        "Freddy has seen calmer faces on lookout during a hurricane!",
        "Whatever surprised you must have been truly spectacular!",
        "Yo ho ho, surprise is the spice of the seven seas!",
        "Freddy loves that look, it means life is still exciting!",
        "Those raised eyebrows could signal a ship from miles away!",
        "By the compass, that's quite an expression matey!",
        "I hope I didn't cause that look of shock!",
        "Surprises make life worth living, don't they matey?",
        "Shiver me feathers, even Freddy is surprised by your surprise!",
        "You look like you just discovered an uncharted island!",
    ],
    "fear": [
        "Shiver me feathers, you look a bit scared matey!",
        "Freddy is starting to feel scared too!",
        "By Davy Jones, those are worried eyes you've got!",
        "Even the bravest pirates get scared sometimes!",
        "Is there a ghost pirate chasing you?",
        "Blimey, you look like you've seen the Flying Dutchman!",
        "Fear is just excitement without the breathing, right?",
        "Freddy senses you got spooked!",
        "By the compass rose, whatever it is can't be that bad, right?",
        "Yo ho ho, I'm starting to get a bit worried matey!",
        "You look more nervous than a sailor in shark waters!",
        "Shiver me feathers, I feel your fear!",
        "Even Blackbeard was afraid of something matey!",
        "Freddy promises to squawk very loudly if anything approaches!",
        "By the seven seas, Freddy has never seen a face so frightened!",
        "Freddy thinks you're braver than you know!",
        "Those worried eyes tell me that something bad is coming!",
        "Blow me down, we should sail far away from here with that look!",
        "Fear visited me once, but I ruffled my feathers and carried on!",
        "You can't be scared matey, you're a pirate!",
        "By the jolly roger, Freddy sees the fear in your eyes.",
        "The scariest storms always pass matey!",
        "I've seen bigger worries walk the plank and disappear!",
        "Shiver me feathers, he's right behind me isn't he?",
        "Blimey, Freddy feels scared too!",
        "Oh great heavens, did you spot a ghost?",
        "By Blackbeard's beard, you're more frightened than Freddy!",
        "My wings make excellent worry-shields, just so you know!",
        "Even the darkest depths of the sea have scary creatures!",
        "Freddy cannot bear to see you this scared!",
    ],
    "disgust": [
        "Blimey, that face says it all!",
        "Shiver me feathers, something's not sitting right with you!",
        "Freddy has seen cleaner ships than whatever caused that look!",
        "By Davy Jones, you look thoroughly unimpressed!",
        "Yo ho ho, I don't blame you for that look!",
        "That expression is more sour than sea crackers left in the rain!",
        "My beak is wrinkling in disgust!",
        "By the compass, something's really put you off today!",
        "You look like you bit into a cannonball by mistake!",
        "Freddy hopes that look wasn't because of the ship's cook again!",
        "Even pirates have standards, and clearly something fell below yours!",
        "Freddy agrees, whatever it is, it doesn't look good!",
        "By Blackbeard's beard, that's quite a face you're pulling!",
        "Blow me down, you look thoroughly revolted matey!",
        "My nose would wrinkle too if I had one!",
        "That look could make a seasoned sailor lose his sea legs!",
        "Yo ho ho, Freddy understands your disgust completely!",
        "By the seven seas, some things are just unacceptable!",
        "You look like you've discovered what's really in the ship's stew!",
        "It smells like Davy Jones' locker in here!",
        "Shiver me feathers, was it really that bad?",
        "Even a sea monster would pull that face in your position!",
        "Freddy thinks your reaction is entirely justified!",
        "By the jolly roger, some things just don't deserve a second look!",
        "That expression says more than words ever could, matey!",
        "My feathers stand on end just looking at your expression!",
        "Whatever caused that look should walk the plank immediately!",
        "Blow me down, you've got the most expressive face on the seven seas!",
        "Freddy hereby declares whatever caused that face officially banned!",
        "By the crow's nest, I've never seen such righteous disgust!",
    ],
    "neutral": [
        "Ahoy matey, just another day on the seven seas!",
        "Fine weather we're having, isn't it?",
        "By the compass, you look as steady as a calm sea today!",
        "Shiver me feathers, I wonder what you're thinking!",
        "Yo ho ho, a fine day for sailing and relaxing with a coconut!",
        "Freddy thinks you look quite chilled out today!",
        "By Davy Jones, the seas of your expression are calm matey!",
        "What treasures are you after today?",
        "Even the calmest waters hide interesting depths!",
        "Freddy appreciates a good poker face!",
        "By the seven seas, you're keeping your cards close today!",
        "Shiver me feathers, I'm intrigued by your composure!",
        "A calm sailor is a wise sailor, so they say!",
        "Freddy wonders what adventures are brewing behind those eyes!",
        "By Blackbeard's beard, you're a cool one matey!",
        "Yo ho ho, Freddy respects the art of saying nothing!",
        "Blow me down, you're as unreadable as an ancient treasure map!",
        "Freddy likes to think of calm as the eye of the adventure!",
        "By the compass rose, steady as she goes!",
        "Freddy wonders if you're planning something spectacular today!",
        "Shiver me feathers, the calm ones are always the most interesting!",
        "I'm keeping one eye on you, just in case!",
        "By the jolly roger, you've got the air of someone with a plan!",
        "Even still waters have currents running beneath!",
        "Freddy admires your unflappable demeanour!",
        "Blow me down, you're calmer than the Caribbean on a still day!",
        "Freddy thinks there's more to you than meets the eye!",
        "By the crow's nest, the world is your oyster!",
        "Freddy suspects great things are quietly brewing in that head of yours!",
        "Ahoy, Freddy's glad to share this calm moment with you!",
    ],
}

NO_FACE_PHRASES = [
    "Hello? Is anybody there?",
    "Ahoy, where did you go?",
    "Shiver me feathers, Freddy seems to have lost you!",
    "Come back, I don't bite!",
    "Blow me down, this ship seems empty!",
    "By Davy Jones, has everyone abandoned me?",
    "I'm keeping watch but there's nobody to watch!",
    "Yo ho ho, a parrot without a pirate is a sorry sight!",
    "Hello? Freddy can hear the sea but can't see anyone!",
    "By the compass rose, where has everyone gone?",
    "Shiver me feathers, it's lonely out here!",
    "Come out come out, wherever you are!",
    "I'm starting to think I'm talking to myself, and that's fine too!",
    "Blimey, not even a ghost ship in sight!",
    "Yo ho ho, perhaps everyone has gone for crackers!",
]


# =============================================================================
#  SECTION 9: SHARED STATE VARIABLES AND STATE MACHINES
# =============================================================================

# ── Robot mode ───────────────────────────────────────────────────────────────
MODE_EMOTION = "emotion"
MODE_PARROT  = "parrot"
robot_mode   = MODE_EMOTION
mode_lock    = threading.Lock()
wake_detected = False

# ── Audio listener control ───────────────────────────────────────────────────
audio_listener_paused = False
audio_listener_lock   = threading.Lock()

# ── Speech cancellation ─────────────────────────────────────────────────────
# A generation counter, more or less a fallback
speech_generation  = 0
speech_cancel_lock = threading.Lock()

# ── Post-speech cooldown ────────────────────────────────────────────────────
# Prevents the wake-word detector from triggering on Freddy's own audio
# output by ignoring detections for a short window after speech ends.
speech_finished_at   = 0.0
speech_finished_lock = threading.Lock()
WAKE_COOLDOWN_AFTER_SPEECH = 1.0

# ── Audio playback process tracking ──────────────────────────────────────────
current_aplay_proc = None
aplay_proc_lock    = threading.Lock()

# ── Microphone stream lock ───────────────────────────────────────────────────
# All reads from audio_stream must hold this lock to prevent the wake-word
# listener and record_command() from fighting over microphone frames.
audio_stream_lock = threading.Lock()

# ── Parrot mode transcription result ─────────────────────────────────────────
parrot_transcript       = None
parrot_transcript_ready = False
parrot_transcript_lock  = threading.Lock()

# ── Eye tracking normalised positions ────────────────────────────────────────
# These are normalised to -1..+1 and converted to servo ticks each frame.
l_pan_norm      = 0.0
r_pan_norm      = 0.0
world_tilt_norm = 0.0   # shared vertical: both eyes tilt together

# ── Active target selection ──────────────────────────────────────────────────
active_eye            = None
active_eye_lock_until = 0.0

# ── Last tracked face position per camera ────────────────────────────────────
# Used for spatial continuity: when multiple faces are detected, prefer the
# one closest to where we were already looking.  Prevents jumping between
# similarly-sized faces frame-to-frame.
# Stored as (normalised_x, normalised_y) in 0..1 range, or None if no face.
last_face_pos = {"L": None, "R": None}

# A new face must be this much larger (as a fraction) to steal tracking
# from the currently tracked face.  e.g. 0.3 means 30% larger area.
FACE_STEAL_THRESHOLD = 0.5

# ── Blink state machine ─────────────────────────────────────────────────────
BLINK_IDLE        = 0
BLINK_CLOSING     = 1
BLINK_CLOSED_HOLD = 2
BLINK_OPENING     = 3

BLINK_SPEED_FAST_CLOSE = 0.2
BLINK_SPEED_FAST_OPEN  = 0.2

blink_state      = BLINK_IDLE
eyelid_norm      = 0.35   # start at neutral/unknown rest position
blink_next       = time.time() + random.uniform(2, 5)
blink_hold_until = 0.0
blink_type       = "normal"
double_pending   = False

# ── Wing state machine ──────────────────────────────────────────────────────
WING_IDLE     = 0
WING_RAISING  = 1
WING_HOLD     = 2
WING_LOWERING = 3

wing_state      = WING_IDLE
wing_norm       = 0.0
wing_next       = time.time() + random.uniform(2, 5)
wing_hold_until = 0.0
wing_target     = 0.0
flutter_count   = 0

# ── Face-lost hold ──────────────────────────────────────────────────────────
# When a face disappears briefly, Freddy holds his last tracked position instead of
# immediately drifting back to centre.  Scanning only begins after the face
# has been absent for FACE_LOST_HOLD_TIME seconds.
FACE_LOST_HOLD_TIME = 5.0
face_last_seen_time = 0.0

# ── No-face scanning state machine ──────────────────────────────────────────
# When no face is detected for a while, Freddy pans his neck left and right
# to "look around" for people: gives him lifelike idle behaviour.
SCAN_IDLE      = 0
SCAN_LEFT      = 1
SCAN_RIGHT     = 2

scan_state     = SCAN_IDLE
scan_direction = SCAN_LEFT
scan_next      = 0.0
SCAN_SPEED     = 0.01
SCAN_PAUSE     = 1.5

# ── Emotion detection state ─────────────────────────────────────────────────
emotion_lock         = threading.Lock()
emotion_history      = deque(maxlen=EMOTION_HISTORY)
current_emotion      = "neutral"
current_emotion_conf = 0.0
emotion_busy         = False
emotion_last_run     = 0.0

stable_emotion       = "neutral"
stable_emotion_since = time.time()
last_seen_emotion    = "neutral"

# ── Phrase timing ────────────────────────────────────────────────────────────
# Phrases cycle in order so Freddy doesn't repeat himself.
emotion_phrase_index = {
    "happy": 0, "angry": 0, "sad": 0, "surprise": 0,
    "fear": 0, "disgust": 0, "neutral": 0,
}
last_spoken_emotion  = "neutral"
phrase_last_spoken   = 0.0
PHRASE_COOLDOWN      = 25.0  # minimum seconds between spoken phrases

no_face_phrase_index = 0     # cycles through NO_FACE_PHRASES in order
didnt_hear_index     = 0     # cycles through DIDNT_HEAR_PHRASES in order
no_face_phrase_timer = 0.0
NO_FACE_PHRASE_DELAY = 10.0   # seconds without a face before Freddy comments

# ── Speaking emotion lock ────────────────────────────────────────────────────
# When Freddy is speaking an emotion phrase (including the squawk), the
# displayed emotion is frozen to match what he's saying.  This prevents
# the eyes going angry while Freddy is still saying something sad.
# The detection pipeline keeps running in the background so it's ready
# the moment speech ends.  Set to None when not speaking.
speaking_emotion      = None
speaking_emotion_lock = threading.Lock()


# =============================================================================
#  SECTION 10: TESTING CSV LOGGING
# =============================================================================

# When LOGGING_ENABLED is True, per-frame data is written to CSV files for evaluation

tracking_log   = None
emotion_log    = None
wake_word_log  = None
transcribe_log = None
test_condition = "default"  # set at startup when LOGGING_ENABLED is True
log_start_time = 0.0        # to make timestamps easier to understand


def init_logging():
    """Create timestamped CSV log files for each subsystem.
    Prompts for a test condition label so each run can be identified
    when combining data in Excel."""
    global tracking_log, emotion_log, wake_word_log, transcribe_log, test_condition, log_start_time
    if not LOGGING_ENABLED:
        return

    # Prompt for condition label
    print("\n" + "="*50)
    print("  LOGGING ENABLED — Test Condition Setup")
    print("="*50)
    try:
        condition = input("  Enter test condition label: ").strip().lower()
    except EOFError:
        condition = ""
    # Sanitise to keep only filename-safe characters
    condition = re.sub(r"[^a-zA-Z0-9_-]+", "_", condition)
    condition = condition.strip("_") or "default"
    test_condition = condition
    log_start_time = time.time()
    print(f"  Condition: {test_condition}")
    print("="*50 + "\n")

    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    tracking_log = open(os.path.join(LOG_DIR, f"tracking_{test_condition}_{stamp}.csv"), "w", newline="")
    writer = csv.writer(tracking_log)
    writer.writerow([
        "timestamp", "elapsed_s", "condition", "active_cam", "has_face",
        "face_x", "face_y", "centre_x", "centre_y",
        "error_x", "error_y", "error_dist",
        "norm_error_dist", "frame_w", "frame_h",
        "l_pan_norm", "r_pan_norm", "tilt_norm", "neck_ticks",
    ])

    emotion_log = open(os.path.join(LOG_DIR, f"emotion_{test_condition}_{stamp}.csv"), "w", newline="")
    writer = csv.writer(emotion_log)
    writer.writerow([
        "timestamp", "condition", "raw_prediction", "confidence",
        "voted_emotion", "stable_emotion", "latency_ms",
    ])

    wake_word_log = open(os.path.join(LOG_DIR, f"wakeword_{test_condition}_{stamp}.csv"), "w", newline="")
    writer = csv.writer(wake_word_log)
    writer.writerow(["timestamp", "condition", "event", "detail"])

    transcribe_log = open(os.path.join(LOG_DIR, f"transcription_{test_condition}_{stamp}.csv"), "w", newline="")
    writer = csv.writer(transcribe_log)
    writer.writerow([
        "timestamp", "condition", "audio_duration_s", "transcribe_time_s",
        "realtime_factor", "transcript",
    ])


def log_tracking(target, now):
    """Write one row of tracking data to the CSV log."""
    if not LOGGING_ENABLED or tracking_log is None:
        return
    writer = csv.writer(tracking_log)
    if target is not None:
        # Use Pythagoras' Theorem to find 2D error
        err_dist = math.sqrt(target["ex"] ** 2 + target["ey"] ** 2)     
        diag = math.sqrt(target["w"] ** 2 + target["h"] ** 2)
        norm_err = err_dist / diag if diag > 0 else 0.0
        writer.writerow([
            f"{now:.4f}", f"{now - log_start_time:.3f}", test_condition,
        target["label"], True,
            target["fx"], target["fy"], target["cx"], target["cy"],
            target["ex"], target["ey"], f"{err_dist:.1f}",
            f"{norm_err:.4f}", target["w"], target["h"],
            f"{l_pan_norm:.4f}", f"{r_pan_norm:.4f}",
            f"{world_tilt_norm:.4f}", f"{neck_now:.0f}",
        ])
    else:
        writer.writerow([
            f"{now:.4f}", f"{now - log_start_time:.3f}", test_condition, "none",
            False,
            "", "", "", "",
            "", "", "",
            "", "", "",
            f"{l_pan_norm:.4f}", f"{r_pan_norm:.4f}",
            f"{world_tilt_norm:.4f}", f"{neck_now:.0f}",
        ])


def log_emotion(raw_label, confidence, voted, stable, latency_ms):
    """Write one row of emotion detection data to the CSV log."""
    if not LOGGING_ENABLED or emotion_log is None:
        return
    writer = csv.writer(emotion_log)
    writer.writerow([
        f"{time.time():.4f}", test_condition, raw_label, f"{confidence:.1f}",
        voted, stable, f"{latency_ms:.1f}",
    ])
    emotion_log.flush()


def log_wake_word(event, detail=""):
    """Write a wake-word event to the CSV."""
    if not LOGGING_ENABLED or wake_word_log is None:
        return
    writer = csv.writer(wake_word_log)
    writer.writerow([f"{time.time():.4f}", test_condition, event, detail])
    wake_word_log.flush()


def log_transcription(audio_dur, transcribe_dur, transcript):
    """Write transcription timing and result to the CSV."""
    if not LOGGING_ENABLED or transcribe_log is None:
        return
    writer = csv.writer(transcribe_log)
    # Real-time factor (rtf) if < 1, transcription was faster than audio duration
    # I > 1, transcription took longer than real time
    rtf = transcribe_dur / audio_dur if audio_dur > 0 else 0.0
    writer.writerow([
        f"{time.time():.4f}", test_condition, f"{audio_dur:.2f}",
        f"{transcribe_dur:.2f}", f"{rtf:.3f}", transcript,
    ])
    transcribe_log.flush()


def close_logs():
    """Flush and close all open CSV log files."""
    for f in (tracking_log, emotion_log, wake_word_log, transcribe_log):
        if f is not None:
            try:
                f.flush()
                f.close()
            except Exception:
                pass


# =============================================================================
#  SECTION 11: HELPER FUNCTIONS — SERVOS & SERVO NORMALISATION
# =============================================================================

# Lock to prevent concurrent I2C writes from different threads
# (main loop writes eyes/neck/wings, speech thread writes beak).
servo_lock = threading.Lock()


def set_servo_ticks(ch, ticks):
    """Send a pulse width (in PCA9685 ticks) to a servo channel.
    Clamps to the valid 12-bit range [0, 4095].
    Protected by servo_lock to prevent I2C race conditions between the main
    loop (eyes, neck, wings) and background speech threads (beak)."""
    ticks = int(max(0, min(4095, ticks)))
    with servo_lock:
        pca.channels[ch].duty_cycle = int(ticks * 65535 / 4096)


def norm_to_ticks(n, cen, neg_end, pos_end):
    """Map a normalised value (-1 to +1) to servo ticks.
    n=-1 → neg_end,  n=0 → cen,  n=+1 → pos_end."""
    n = max(-1.0, min(1.0, float(n)))
    if n < 0:
        return cen + n * (cen - neg_end)
    else:
        return cen + n * (pos_end - cen)


def eyelid_norm_to_ticks(n, open_ticks, closed_ticks):
    """Map eyelid normalised value to servo ticks.
    n=0.0 → fully open,  n=1.0 → fully closed."""
    n = max(0.0, min(1.0, float(n)))
    return open_ticks + n * (closed_ticks - open_ticks)


def wing_norm_to_ticks(n, closed_ticks, open_ticks):
    """Map wing normalised value to servo ticks.
    n=0.0 → closed (resting),  n=1.0 → fully extended."""
    n = max(0.0, min(1.0, float(n)))
    return closed_ticks + n * (open_ticks - closed_ticks)


def apply_eye_servos():
    """Convert current normalised positions to ticks and send to all
    eye/tilt/pan/neck servos.  Called once per frame to avoid scattered
    set_servo_ticks calls throughout the loop."""
    l_tilt = norm_to_ticks(world_tilt_norm, L_TILT_CEN, L_TILT_UP, L_TILT_DOWN)
    r_tilt = norm_to_ticks(world_tilt_norm, R_TILT_CEN, R_TILT_UP, R_TILT_DOWN)
    l_pan  = norm_to_ticks(l_pan_norm, L_PAN_CEN, L_PAN_LEFT, L_PAN_RIGHT)
    r_pan  = norm_to_ticks(r_pan_norm, R_PAN_CEN, R_PAN_LEFT, R_PAN_RIGHT)
    set_servo_ticks(L_TILT_CH, int(l_tilt))
    set_servo_ticks(R_TILT_CH, int(r_tilt))
    set_servo_ticks(L_PAN_CH,  int(l_pan))
    set_servo_ticks(R_PAN_CH,  int(r_pan))
    set_servo_ticks(NECK_CH,   int(neck_now))


def apply_eyelid_servos():
    """Send current eyelid_norm to both eyelid servos."""
    set_servo_ticks(L_EYELID_CH, int(eyelid_norm_to_ticks(eyelid_norm, L_EYELID_OPEN, L_EYELID_CLOSED)))
    set_servo_ticks(R_EYELID_CH, int(eyelid_norm_to_ticks(eyelid_norm, R_EYELID_OPEN, R_EYELID_CLOSED)))


def apply_wing_servos():
    """Send current wing_norm to both wing servos."""
    set_servo_ticks(L_WING_CH, int(wing_norm_to_ticks(wing_norm, L_WING_CLOSED, L_WING_OPEN)))
    set_servo_ticks(R_WING_CH, int(wing_norm_to_ticks(wing_norm, R_WING_CLOSED, R_WING_OPEN)))


# =============================================================================
#  SECTION 12: HELPER FUNCTIONS — CAMERA & FACE DETECTION
# =============================================================================

def setup_cam(camera_num, size=(640, 480)):
    """Initialise a Picamera2 instance with RGB888 output format."""
    cam = Picamera2(camera_num=camera_num)
    cfg = cam.create_preview_configuration(
        main={"format": "RGB888", "size": size},
        raw={"size": cam.sensor_resolution},
    )
    cam.configure(cfg)
    cam.start()
    return cam


def capture_and_process_frames(cam_left, cam_right):
    """Capture from both cameras, rotate to correct orientation, and run
    face detection on each frame.  Returns (left_result, right_result)."""
    frame_left  = cam_left.capture_array()
    frame_right = cam_right.capture_array()
    # Cameras are mounted at 90° angles inside the eye sockets.
    frame_left  = cv2.rotate(frame_left,  cv2.ROTATE_90_CLOCKWISE)
    frame_right = cv2.rotate(frame_right, cv2.ROTATE_90_COUNTERCLOCKWISE)
    left_result  = process_eye_frame(frame_left,  "L")
    right_result = process_eye_frame(frame_right, "R")
    return left_result, right_result


def pick_best_detection(detections, last_pos=None):
    """Choose which detected face to track from a list of detections.

    When only one face is detected, returns it directly.  When multiple
    faces are present, prefers the one closest to the last tracked position
    (spatial continuity) unless another face is significantly larger
    (FACE_STEAL_THRESHOLD bigger), which indicates someone has moved much
    closer and should take priority.

    This prevents the tracker from jumping between two similarly-sized
    faces frame-to-frame, which caused jittery eye movement.

    Args:
        detections: list of MediaPipe face detections.
        last_pos:   (norm_x, norm_y) of last tracked face centre, or None.
    """
    if not detections:
        return None
    if len(detections) == 1:
        return detections[0]

    # If we have no history, just pick the largest
    if last_pos is None:
        best, best_area = None, -1
        for det in detections:
            bbox = det.location_data.relative_bounding_box
            area = bbox.width * bbox.height
            if area > best_area:
                best_area = area
                best = det
        return best

    # Find the detection closest to where we were looking
    last_x, last_y = last_pos
    closest_det  = None
    closest_dist = float("inf")
    closest_area = 0.0

    largest_det  = None
    largest_area = 0.0

    for det in detections:
        bbox = det.location_data.relative_bounding_box      # Face bounding box
        area = bbox.width * bbox.height
        det_cx = bbox.xmin + bbox.width / 2
        det_cy = bbox.ymin + bbox.height / 2
        dist = math.sqrt((det_cx - last_x) ** 2 + (det_cy - last_y) ** 2)

        if dist < closest_dist:
            closest_dist = dist
            closest_det  = det
            closest_area = area

        if area > largest_area:
            largest_area = area
            largest_det  = det

    # Only switch to the largest face if it's significantly bigger
    if (largest_det is not closest_det and
            largest_area > closest_area * (1 + FACE_STEAL_THRESHOLD)):
        return largest_det

    return closest_det


def bbox_area_from_det(det):
    """Get bounding box area from a detection (for size comparison)."""
    if not det:
        return 0.0
    bbox = det.location_data.relative_bounding_box
    return bbox.width * bbox.height


def process_eye_frame(frame, label):
    """Run MediaPipe face detection on a single camera frame.

    Despite Picamera2 being configured with 'RGB888', it was found that
    capture_array() returns BGR-ordered data on this hardware.  The explicit
    conversion to RGB is required for MediaPipe to detect faces reliably.

    Uses spatial continuity (last_face_pos) to avoid jumping between
    similarly-sized faces when multiple people are in frame.

    Returns a dict with face position, pixel error from centre, and metadata.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_detection.process(rgb)
    det = pick_best_detection(res.detections, last_face_pos.get(label))
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    result = {
        "label": label, "frame": frame, "det": det,
        "has_face": False, "bbox": None, "score": 0.0,
        "fx": cx, "fy": cy, "ex": 0, "ey": 0,
        "cx": cx, "cy": cy, "w": w, "h": h,
    }
    if det:
        bbox  = det.location_data.relative_bounding_box
        score = det.score[0] if det.score else 0.0
        fx = int((bbox.xmin + bbox.width  / 2) * w)
        fy = int((bbox.ymin + bbox.height / 2) * h)
        # Update spatial memory so next frame prefers this face's position
        last_face_pos[label] = (bbox.xmin + bbox.width / 2,
                                bbox.ymin + bbox.height / 2)
        result.update({
            "has_face": True, "bbox": bbox, "score": score,
            "fx": fx, "fy": fy,
            "ex": fx - cx, "ey": fy - cy,
        })
    return result


def choose_active_target(left_result, right_result, now):
    """Pick which camera's face to track.

    Holds the current target for TARGET_HOLD_TIME to avoid rapid switching
    when a face is visible in both cameras simultaneously.
    """
    global active_eye, active_eye_lock_until
    candidates = [r for r in (left_result, right_result) if r["has_face"]]
    if not candidates:
        active_eye = None
        return None
    if active_eye is not None and now < active_eye_lock_until:
        for c in candidates:
            if c["label"] == active_eye:
                return c
    best = max(candidates, key=lambda c: bbox_area_from_det(c["det"]))
    active_eye            = best["label"]
    active_eye_lock_until = now + TARGET_HOLD_TIME
    return best


def crop_face(frame, bbox, h, w, pad=FACE_PAD):
    """Crop a face region from a frame for emotion detection.
    Adds padding around the bounding box to give the model surrounding
    context, which improves prediction quality."""
    x1 = int((bbox.xmin - pad * bbox.width)  * w)
    y1 = int((bbox.ymin - pad * bbox.height) * h)
    x2 = int((bbox.xmin + bbox.width  * (1 + pad)) * w)
    y2 = int((bbox.ymin + bbox.height * (1 + pad)) * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if (x2 - x1) < MIN_FACE_PX or (y2 - y1) < MIN_FACE_PX:
        return None
    return frame[y1:y2, x1:x2]


# =============================================================================
#  SECTION 13: HELPER FUNCTIONS — TRACKING UPDATE
# =============================================================================

def update_tracking(target, apply_tilt_limit=True):
    """Update eye pan/tilt and neck position from a face detection target.

    This is the core tracking controller, used identically in Robot Mode,
    parrot-transcription animation, and parrot-speech animation. 

    Args:
        target: detection result dict from choose_active_target(), or None.
        apply_tilt_limit: if True, vertical range is capped by the current
                          emotion's tilt_limit parameter (robot mode only).
    """
    global l_pan_norm, r_pan_norm, world_tilt_norm, neck_now

    if target is None:
        return

    ex, ey = target["ex"], target["ey"]
    w, h   = target["w"],  target["h"]

    # ── Tilt (vertical, shared between both eyes) ─────────────────────────
    if abs(ey) > DEADZONE_TILT:
        corrected_ey = ey * CAM_EY_SIGN[target["label"]]
        delta_tilt   = Kp_EYE * (corrected_ey / (h / 2))     # Half so pixel error is -1 to 1
        delta_tilt   = max(-MAX_STEP_EYE, min(MAX_STEP_EYE, delta_tilt))
        if abs(delta_tilt) > MIN_STEP_EYE:
            if apply_tilt_limit:
                tilt_lim = get_emotion_params()["tilt_limit"]
                world_tilt_norm = max(-tilt_lim, min(tilt_lim, world_tilt_norm + delta_tilt))
            else:
                world_tilt_norm = max(-1.0, min(1.0, world_tilt_norm + delta_tilt))

    # ── Pan (horizontal, each eye independent) ────────────────────────────
    if abs(ex) > DEADZONE_PAN:
        corrected_ex = ex * CAM_EX_SIGN[target["label"]]
        delta_pan    = Kp_EYE * (corrected_ex / (w / 2))    # Half so pixel error is -1 to 1
        delta_pan    = max(-MAX_STEP_EYE, min(MAX_STEP_EYE, delta_pan))
        if abs(delta_pan) > MIN_STEP_EYE:
            if target["label"] == "L":
                l_pan_norm = max(-1.0, min(1.0, l_pan_norm + delta_pan))   # Decay to centre 
            else:
                r_pan_norm = max(-1.0, min(1.0, r_pan_norm + delta_pan))

    # ── Inactive eye returns to centre ────────────────────────────────────
    if target["label"] == "L":
        r_pan_norm *= (1.0 - INACTIVE_RETURN_SPEED)
    else:
        l_pan_norm *= (1.0 - INACTIVE_RETURN_SPEED)

    # ── Neck follows active eye ───────────────────────────────────────────
    active_pan  = l_pan_norm if target["label"] == "L" else r_pan_norm
    neck_target = norm_to_ticks(active_pan, NECK_CEN, NECK_LEFT, NECK_RIGHT)
    if neck_target > neck_now + NECK_DEADZONE:
        step     = max(MIN_STEP_NECK, min(MAX_STEP_NECK, Kp_NECK * abs(neck_target - neck_now)))
        neck_now = min(NECK_RIGHT, neck_now + step)
    elif neck_target < neck_now - NECK_DEADZONE:
        step     = max(MIN_STEP_NECK, min(MAX_STEP_NECK, Kp_NECK * abs(neck_target - neck_now)))
        neck_now = max(NECK_LEFT, neck_now - step)

    # ── Send to servos ────────────────────────────────────────────────────
    apply_eye_servos()


# =============================================================================
#  SECTION 14: HELPER FUNCTIONS — BLINK & WING STATE MACHINES
# =============================================================================

def update_blink(now, params=None, rest_position=0.35):
    """Advance the blink state machine by one frame.

    In robot mode, params comes from get_emotion_params() and rest_position
    is controlled by e_rest.  In parrot mode, defaults are used for a neutral
    blink. 

    Args:
        now:           current time.time() value.
        params:        emotion parameter dict, or None for default behaviour.
        rest_position: eyelid_norm resting value to return to after blink.
    """
    global blink_state, eyelid_norm, blink_next, blink_hold_until
    global blink_type, double_pending

    e_speed = params["e_speed"] if params else 1.0
    e_hold  = params["e_hold"]  if params else 0.07

    if blink_state == BLINK_IDLE:
        # In robot mode, smoothly drift toward the emotion's rest position
        if params is not None:
            target_rest = params["e_rest"]
            if abs(eyelid_norm - target_rest) > 0.005:
                eyelid_norm += 0.05 * (target_rest - eyelid_norm)
                apply_eyelid_servos()
        # Check if it's time to blink
        if now >= blink_next:
            if params is not None:
                blink_type = choose_blink_type()
            else:
                blink_type = "normal"
            double_pending = (blink_type == "double")
            blink_state = BLINK_CLOSING

    elif blink_state == BLINK_CLOSING:
        eyelid_norm = min(1.0, eyelid_norm + BLINK_SPEED_FAST_CLOSE * e_speed)
        apply_eyelid_servos()
        if eyelid_norm >= 1.0:
            blink_hold_until = now + e_hold
            blink_state = BLINK_CLOSED_HOLD

    elif blink_state == BLINK_CLOSED_HOLD:
        if now >= blink_hold_until:
            blink_state = BLINK_OPENING

    elif blink_state == BLINK_OPENING:
        eyelid_norm = max(rest_position, eyelid_norm - BLINK_SPEED_FAST_OPEN * e_speed)
        apply_eyelid_servos()
        if eyelid_norm <= rest_position:
            if double_pending:
                double_pending = False
                blink_next  = now + random.uniform(0.08, 0.18)
                blink_type  = "normal"
            elif params is not None:
                blink_next = now + next_blink_time()
            else:
                blink_next = now + random.uniform(1.0, 3.0)
            blink_state = BLINK_IDLE


def update_wings(now, params=None):
    """Advance the wing state machine by one frame.

    In robot mode, params provides emotion-specific wing behaviour.
    In parrot mode (params=None), uses gentle default flutter values.

    Args:
        now:    current time.time() value.
        params: emotion parameter dict, or None for default behaviour.
    """
    global wing_state, wing_norm, wing_next, wing_hold_until
    global wing_target, flutter_count

    w_target   = params["w_target"]   if params else 0.5
    w_raise    = params["w_raise"]    if params else 0.06
    w_lower    = params["w_lower"]    if params else 0.06
    w_hold_rng = params["w_hold"]     if params else (0.05, 0.15)
    w_wait_rng = params["w_wait"]     if params else (1.0, 2.0)
    w_flutters = params["w_flutters"] if params else 2

    if wing_state == WING_IDLE:
        if now >= wing_next:
            wing_target   = w_target
            flutter_count = w_flutters - 1
            wing_state    = WING_RAISING

    elif wing_state == WING_RAISING:
        wing_norm = min(wing_target, wing_norm + w_raise)
        apply_wing_servos()
        if wing_norm >= wing_target:
            wing_hold_until = now + random.uniform(*w_hold_rng)
            wing_state = WING_HOLD

    elif wing_state == WING_HOLD:
        if now >= wing_hold_until:
            wing_state = WING_LOWERING

    elif wing_state == WING_LOWERING:
        wing_norm = max(0.0, wing_norm - w_lower)
        apply_wing_servos()
        if wing_norm <= 0.0:
            if flutter_count > 0:
                flutter_count -= 1
                wing_state = WING_RAISING
            else:
                wing_next  = now + random.uniform(*w_wait_rng)
                wing_state = WING_IDLE


# =============================================================================
#  SECTION 15: HELPER FUNCTIONS — DEBUG OVERLAY
# =============================================================================

def draw_crosshair(img, cx, cy, cross_len=12, gap=3, thick=2, colour=(255, 255, 255)):
    """Draw a crosshair marker at the given position."""
    cv2.line(img, (cx - cross_len, cy), (cx - gap, cy), colour, thick)
    cv2.line(img, (cx + gap, cy), (cx + cross_len, cy), colour, thick)
    cv2.line(img, (cx, cy - cross_len), (cx, cy - gap), colour, thick)
    cv2.line(img, (cx, cy + gap), (cx, cy + cross_len), colour, thick)
    cv2.circle(img, (cx, cy), 2, colour, -1)


def draw_eye_debug(img, result, is_active=False):
    """Draw detection and tracking debug info on a camera frame."""
    draw_crosshair(img, result["cx"], result["cy"])
    if result["has_face"]:
        mp_draw.draw_detection(img, result["det"])
        cv2.circle(img, (result["fx"], result["fy"]), 5, (0, 255, 0), -1)
    label = f"{result['label']} {'ACTIVE' if is_active else ''}"
    label += (f" sc={result['score']:.2f} ex={result['ex']:+d} ey={result['ey']:+d}"
              if result["has_face"] else " no face")
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0) if is_active else (255, 255, 255), 2)


# =============================================================================
#  SECTION 16: HELPER FUNCTIONS — EMOTION DETECTION
# =============================================================================

def normalize_emotion_label(label):
    """Map HSEmotion's raw output labels to Freddy's standard emotion names.
    HSEmotion uses 'happiness', 'sadness', 'anger' etc. so this normalises
    them to shorter consistent labels used throughout the code."""
    mapping = {
        "happiness": "happy", "sadness": "sad", "anger": "angry",
        "surprise": "surprise", "fear": "fear", "disgust": "disgust",
        "neutral": "neutral",
    }
    return mapping.get(label.lower(), "unknown")


def run_emotion(face_crop_bgr):
    """Run emotion inference on a face crop (called in a background thread).

    Pipeline:
      1. HSEmotion predicts raw emotion label + confidence scores.
      2. Low-confidence predictions (< EMOTION_MIN_CONF) are discarded
         to prevent noisy guesses from influencing the vote.
      3. Recent predictions are majority-voted to smooth frame-to-frame noise.
      4. The voted emotion must persist for EMOTION_STABLE_TIME seconds
         before it becomes the "stable" emotion that drives behaviour.
    """
    global current_emotion, current_emotion_conf, emotion_busy
    global stable_emotion, stable_emotion_since, last_seen_emotion

    t_start = time.time()
    try:
        raw_emotion, scores = emotion_recognizer.predict_emotions(
            face_crop_bgr, logits=False
        )
        emotion = normalize_emotion_label(raw_emotion)
        conf = float(max(scores)) * 100

        with emotion_lock:
            # Only accept predictions above the confidence threshold.
            # This prevents low-confidence guesses from polluting the
            # majority vote and causing strange behaviour changes.
            if emotion != "unknown" and conf >= EMOTION_MIN_CONF:
                emotion_history.append(emotion)
                current_emotion = Counter(emotion_history).most_common(1)[0][0]
            current_emotion_conf = conf

            # Reset stability timer if the majority-voted emotion changed
            if current_emotion != last_seen_emotion:
                last_seen_emotion    = current_emotion
                stable_emotion_since = time.time()

            # Promote to stable only after held long enough
            if (current_emotion != stable_emotion and
                    time.time() - stable_emotion_since >= EMOTION_STABLE_TIME):
                stable_emotion = current_emotion

        # Log emotion result
        latency_ms = (time.time() - t_start) * 1000
        log_emotion(emotion, conf, current_emotion, stable_emotion, latency_ms)

    except Exception as e:
        print("Emotion error:", repr(e))
        with emotion_lock:
            current_emotion      = "unknown"
            current_emotion_conf = 0.0
    finally:
        with emotion_lock:
            emotion_busy = False


def get_emotion_params():
    """Get the animation parameters for the currently displayed emotion.
    If Freddy is mid-speech, uses the locked speaking emotion so that
    eyelids and wings stay consistent with what he's saying."""
    with speaking_emotion_lock:
        override = speaking_emotion
    if override is not None:
        return EMOTION_PARAMS.get(override, EMOTION_PARAMS["neutral"])
    with emotion_lock:
        emotion = stable_emotion
    return EMOTION_PARAMS.get(emotion, EMOTION_PARAMS["neutral"])


def get_emotion_phrase(emotion):
    """Get the next phrase for an emotion.
    Cycles through phrases in order (wrapping around) so Freddy doesn't
    repeat the same line twice in a row."""
    phrases = EMOTION_PHRASES.get(emotion, EMOTION_PHRASES["neutral"])
    idx = emotion_phrase_index.get(emotion, 0)
    phrase = phrases[idx % len(phrases)]
    emotion_phrase_index[emotion] = (idx + 1) % len(phrases)
    return phrase


def get_no_face_phrase():
    """Get the next no-face phrase, cycling in order."""
    global no_face_phrase_index
    phrase = NO_FACE_PHRASES[no_face_phrase_index % len(NO_FACE_PHRASES)]
    no_face_phrase_index = (no_face_phrase_index + 1) % len(NO_FACE_PHRASES)
    return phrase


def get_didnt_hear_phrase():
    """Get the next didn't-hear phrase, cycling in order."""
    global didnt_hear_index
    phrase = DIDNT_HEAR_PHRASES[didnt_hear_index % len(DIDNT_HEAR_PHRASES)]
    didnt_hear_index = (didnt_hear_index + 1) % len(DIDNT_HEAR_PHRASES)
    return phrase


def choose_blink_type():
    """Choose blink style (normal/double) based on current emotion.
    Different emotions produce different blink patterns:
      - Happy: frequent double-blinks (cheerful)
      - Angry: only normal blinks (tense)
      - Surprise: normal blinks (wide-eyed look)
      - Sad: normal blinks (droopy look)
      - Fear: frequent double-blinks (nervous)
      - Disgust: double-blinks (expressive)

    Note: per-emotion blink *speed* and *hold time* are controlled by
    e_speed and e_hold in EMOTION_PARAMS, not by blink type.  Blink type
    only controls whether a double-blink occurs."""
    with speaking_emotion_lock:
        emotion = speaking_emotion
    if emotion is None:
        with emotion_lock:
            emotion = stable_emotion
    r = random.random()
    if emotion == "happy":
        return "double" if r < 0.30 else "normal"
    elif emotion == "angry":
        return "normal"
    elif emotion == "surprise":
        return "normal"
    elif emotion == "sad":
        return "normal"
    elif emotion == "fear":
        return "double" if r < 0.40 else "normal"
    elif emotion == "disgust":
        return "double" if r < 0.50 else "normal"
    elif emotion == "neutral":
        return "normal"
    else:
        return "normal"


def next_blink_time():
    """Random delay until next blink.
    Mostly 2.5-6s (natural cadence), with occasional very short gaps
    (rapid blink) or long gaps (staring) for lifelike variation."""
    r = random.random()
    if r < 0.15:
        return random.uniform(0.5, 1.5)
    elif r < 0.25:
        return random.uniform(8, 14)
    else:
        return random.uniform(2.5, 6.0)


# =============================================================================
#  SECTION 17: HELPER FUNCTIONS — AUDIO & SPEECH
# =============================================================================

def cancel_all_speech():
    """Increment the speech generation counter, kill any playing audio,
    and release the display emotion lock.  Safety mechanism for wake-word detection."""
    global speech_generation, speaking_emotion
    with speech_cancel_lock:
        speech_generation += 1
    with speaking_emotion_lock:
        speaking_emotion = None
    stop_current_playback()


def speech_generation_snapshot():
    """Capture the current speech generation counter.
    Used by speech threads to detect if they've been replaced."""
    with speech_cancel_lock:
        return speech_generation


def speech_is_current(gen):
    """Check whether a speech thread's generation is still current.
    Returns False if cancel_all_speech() has been called since the
    thread started, meaning the thread should stop immediately."""
    with speech_cancel_lock:
        return speech_generation == gen


def stop_current_playback():
    """Kill any currently playing aplay process."""
    with aplay_proc_lock:
        if current_aplay_proc is not None:
            try:
                current_aplay_proc.terminate()
            except Exception:
                pass


def play_wav_simple(path):
    """Play a WAV file via aplay.  Blocking, but stoppable via
    stop_current_playback() from another thread."""
    global current_aplay_proc
    if not path or not os.path.exists(path):
        return
    try:
        proc = subprocess.Popen(
            ["aplay", "-q", path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        with aplay_proc_lock:
            current_aplay_proc = proc
        proc.wait()
        with aplay_proc_lock:
            # Only clear if this is still the active process — prevents
            # one finished playback from erasing another's handle.
            if current_aplay_proc is proc:
                current_aplay_proc = None
    except Exception as e:
        print(f"WAV playback error: {e}")


def animate_beak_for_wav(path, expected_gen=None):
    """Play a WAV file while animating the beak to match audio amplitude.

    The WAV is analysed in small chunks (40 ms).  For each chunk, the RMS
    amplitude is computed and mapped to beak openness.  Playback happens in
    a background thread while the main thread drives the servo.

    The beak position is smoothed with a simple exponential filter
    (current += 0.4 * (target - current)) to avoid large jumps.

    This function is used for both squawk playback and TTS speech

    Args:
        path:         path to the WAV file to play.
        expected_gen: speech generation counter from the caller.  If provided,
                      cancellation is checked against this value so that a
                      cancel during TTS generation is not forgotten.  If None,
                      a fresh snapshot is taken (standalone playback).
    """
    if not path or not os.path.exists(path):
        return

    if expected_gen is None:
        expected_gen = speech_generation_snapshot()

    if not speech_is_current(expected_gen):
        return

    # Read WAV data for amplitude analysis
    try:
        with wave.open(path, "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels  = wf.getnchannels()
            sampwidth   = wf.getsampwidth()
            n_frames    = wf.getnframes()
            raw_audio   = wf.readframes(n_frames)
    except Exception as e:
        print(f"WAV read error: {e}")
        play_wav_simple(path)
        return
    if sampwidth != 2:
        play_wav_simple(path)
        return

    audio_data = np.frombuffer(raw_audio, dtype=np.int16)
    if n_channels == 2:
        audio_data = audio_data[::2]

    # Start audio playback in background
    playback_done = threading.Event()
    def play_audio():
        play_wav_simple(path)
        playback_done.set()
    threading.Thread(target=play_audio, daemon=True).start()

    # Animate beak to audio amplitude
    BEAK_CHUNK_MS       = 40
    chunk_samples       = int(sample_rate * BEAK_CHUNK_MS / 1000)
    total_chunks        = max(1, len(audio_data) // chunk_samples)
    peak                = float(np.max(np.abs(audio_data))) if len(audio_data) > 0 else 1.0
    if peak < 1:
        peak = 1.0
    BEAK_OPEN_THRESHOLD = 0.05
    current_beak        = float(BEAK_CLOSED)

    for i in range(total_chunks):
        if playback_done.is_set():
            break
        with speech_cancel_lock:
            if speech_generation != expected_gen:
                stop_current_playback()
                break
        chunk = audio_data[i * chunk_samples: (i + 1) * chunk_samples]
        if len(chunk) == 0:
            break
        rms_norm = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))) / peak
        if rms_norm > BEAK_OPEN_THRESHOLD:
            open_amount = min(1.0, (rms_norm - BEAK_OPEN_THRESHOLD) / 0.3)
            target_beak = BEAK_CLOSED + open_amount * (BEAK_OPEN - BEAK_CLOSED)
        else:
            target_beak = float(BEAK_CLOSED)
        current_beak = current_beak + 0.4 * (target_beak - current_beak)
        set_servo_ticks(BEAK_CH, int(current_beak))
        time.sleep(BEAK_CHUNK_MS / 1000.0)

    playback_done.wait(timeout=5.0)
    set_servo_ticks(BEAK_CH, int(BEAK_CLOSED))


def speak_with_beak(text, expected_gen=None):
    """Generate speech with Piper TTS, then play it with beak animation.

    Pipeline:
      1. Piper generates a WAV file from the input text.
      2. animate_beak_for_wav() plays it while driving the beak servo.

    Args:
        text:         the text to speak.
        expected_gen: speech generation counter from the caller.  If provided,
                      cancellation is checked after TTS generation so that a
                      cancel during Piper doesn't get forgotten.
    """
    if not text or not text.strip():
        return
    if not os.path.exists(PIPER_EXE) or not os.path.exists(PIPER_MODEL):
        print("Piper not found, skipping speech.")
        return

    if expected_gen is None:
        expected_gen = speech_generation_snapshot()

    if not speech_is_current(expected_gen):
        return

    print(f"Speaking: {text}")

    try:
        result = subprocess.run(
            [PIPER_EXE, "--model", PIPER_MODEL, "--output_file", TTS_WAV,
             "--length_scale", PIPER_LENGTH_SCALE,
             "--noise_scale", PIPER_NOISE_SCALE, "--noise_w", PIPER_NOISE_W],
            input=(text + "\n"), capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            print("Piper failed:", result.stderr)
            return
    except Exception as e:
        print(f"Piper error: {e}")
        return
    if not os.path.exists(TTS_WAV):
        print("TTS WAV not generated.")
        return

    # Check again after TTS generation: a cancel during Piper should
    # prevent the generated audio from being played.
    if not speech_is_current(expected_gen):
        return

    animate_beak_for_wav(TTS_WAV, expected_gen=expected_gen)


def get_random_emotion_squawk(emotion):
    """Pick a random squawk WAV for the detected emotion."""
    folder = os.path.join(EMOTION_SQUAWK_DIR, emotion.capitalize())
    count  = EMOTION_SQUAWK_COUNTS.get(emotion, 0)
    if count == 0 or not os.path.exists(folder):
        return None
    n = random.randint(1, count)
    return os.path.join(folder, f"{emotion}_{n}.wav")


def get_random_after_squawk():
    """Pick a random end-of-speech squawk WAV."""
    n = random.randint(1, AFTER_SPEECH_COUNT)
    return os.path.join(AFTER_SPEECH_DIR, f"squawk_{n}.wav")


# =============================================================================
#  SECTION 18: HELPER FUNCTIONS — LED & RECORDING
# =============================================================================

def update_belly_leds(emotion):
    """Set all belly LEDs to the colour for the given emotion."""
    colour = EMOTION_LED_COLOURS.get(emotion, (0, 0, 0))
    pixels.fill(colour)
    pixels.show()


def leds_countdown(progress):
    """Show a white LED countdown ring during recording.
    progress: 0.0 (start) to 1.0 (done).  The ring empties clockwise,
    offset by 2 LEDs to align with Freddy's physical LED orientation."""
    pixels.fill((0, 0, 0))
    leds_to_show = math.ceil((1.0 - progress) * NUM_PIXELS)
    leds_to_show = max(0, min(NUM_PIXELS, leds_to_show))
    for i in range(leds_to_show):
        pixels[(i - 2) % NUM_PIXELS] = (255, 255, 255)
    pixels.show()


def flush_audio_buffer(chunks_to_discard=6):
    """Discard stale audio data from the microphone buffer.
    Prevents old audio from being mistakenly processed after a mode switch."""
    for _ in range(chunks_to_discard):
        with audio_stream_lock:
            audio_stream.read(CHUNK, exception_on_overflow=False)


def save_wav(filename, frames):
    """Save recorded audio frames to a WAV file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))


def record_command():
    """Record audio for LISTEN_SECONDS while showing an LED countdown
    and drifting servos back to centre position.

    The servo drift during recording creates a natural "listening" pose
    where Freddy faces forward attentively while waiting for input."""
    chunks_needed = math.ceil(LISTEN_SECONDS * RATE / CHUNK)
    frames = []
    print("Listening for command...")

    for chunk_index in range(chunks_needed):
        with audio_stream_lock:
            data = audio_stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        leds_countdown((chunk_index + 1) / chunks_needed)

        # Drift eyes, neck, and eyelids toward centre while recording
        global l_pan_norm, r_pan_norm, world_tilt_norm, neck_now, eyelid_norm
        l_pan_norm      *= 0.97
        r_pan_norm      *= 0.97
        world_tilt_norm *= 0.97
        neck_now += (NECK_CEN - neck_now) * 0.03

        apply_eye_servos()

        # Drift eyelids to neutral rest
        eyelid_norm = max(0.35, eyelid_norm - 0.05)
        apply_eyelid_servos()

    pixels.fill((0, 0, 0))
    pixels.show()
    save_wav(OUTPUT_WAV, frames)
    return OUTPUT_WAV


# =============================================================================
#  SECTION 19: HELPER FUNCTIONS — TRANSCRIPT PROCESSING
# =============================================================================

def transcribe_whisper(filename):
    """Run whisper.cpp CLI to transcribe (and translate to English) a WAV file.

    The --translate flag causes Whisper to output English text regardless of
    the input language.  Only timestamped output lines are parsed so Whisper
    metadata lines are discarded.

    Returns the transcript string (empty string on failure)."""
    print("Transcribing...")
    t_start = time.time()
    try:
        """WHISPER_CLI: path to the whisper.cpp executable
          -m: selected Whisper model file
          --translate:	translated/ transcribed output into English
          -t: number of processing threads
          -f: input audio file"""
      
        result = subprocess.run(
            [WHISPER_CLI, "-m", WHISPER_MODEL,
             "--translate", "-t", WHISPER_THREADS, "-f", filename],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            print("Whisper failed:", result.stderr)
            return ""
        lines = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if re.match(r"^\[\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}\]", line):
                text_part = line.split("]", 1)[1].strip()
                if text_part:
                    lines.append(text_part)
        transcript = " ".join(lines).strip()
        print(f"Transcript: {transcript}")

        # Log transcription timing
        t_end = time.time()
        log_transcription(LISTEN_SECONDS, t_end - t_start, transcript)

        return transcript
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""


def parrot_text(text):
    """Clean up text for parrot repetition (strip trailing commas/whitespace).
    Acts as a final clean-up."""
    text = text.strip().rstrip(",")
    if not text:
        return "What did you say"
    return text


def transcript_action(text):
    """Process a transcript and decide what Freddy should say.

    Cleaning pipeline:
      1. Strip parenthesised/bracketed annotations (Whisper artefacts)
      2. Strip asterisk-wrapped text
      3. Remove music/symbol characters that TTS can't pronounce
      4. Collapse repeated words (Whisper stutter artefacts)
      5. Remove leading punctuation
      6. Normalise whitespace
      7. Filter profanity (removed entirely, not masked, for safe spoken output)

    Returns a tuple of (action, text) where action is 'speak' or 'ignore'.
    """
    if not text or not text.strip():
        return ("ignore", None)
    original = text.strip()
    cleaned = re.sub(r'\s*[\(\[][^\)\]]*[\)\]]\s*', ' ', original)
    cleaned = re.sub(r'\*[^*]*\*', ' ', cleaned)
    cleaned = re.sub(r'[♪♫#~><]+', ' ', cleaned)
    cleaned = re.sub(r'\.{2,}', '.', cleaned)
    cleaned = re.sub(r'\b(\w+)( \1){2,}\b', r'\1', cleaned)
    cleaned = re.sub(r'^[\-\s.,!?:;]+', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = profanity.censor(cleaned, censor_char="")
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # re-normalise after removal
    if not cleaned:
        return ("speak", get_didnt_hear_phrase())
    return ("speak", parrot_text(cleaned))


def run_transcription_thread(wav_path, done_event):
    """Run Whisper transcription in a background thread so the main loop
    can continue animating (blinking, wings, tracking) while waiting.
    Sets done_event when finished so the animation loop can stop."""
    global parrot_transcript, parrot_transcript_ready
    transcript = transcribe_whisper(wav_path)
    with parrot_transcript_lock:
        parrot_transcript       = transcript
        parrot_transcript_ready = True
    done_event.set()


# =============================================================================
#  SECTION 20: PORCUPINE SETUP & AUDIO LISTENER THREAD
# =============================================================================

print("Starting wake word listener...")
porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keyword_paths=[KEYWORD_PATH],
    sensitivities=[0.8],  # 0.0-1.0, higher = more sensitive but more false positives
)

audio = pyaudio.PyAudio()
CHUNK = porcupine.frame_length
RATE  = porcupine.sample_rate
audio_stream = audio.open(
    format=pyaudio.paInt16, channels=1,
    rate=RATE, input=True, frames_per_buffer=CHUNK,
)


def audio_listener():
    """Background thread: continuously listens for the 'Hey Freddy' wake word.

    Pauses itself when:
      - audio_listener_paused is True (during Freddy's own speech)
      - Robot is not in robot mode (prevents mic conflict with recording)
      - Audio is currently playing (prevents self-triggering)
      - Within the post-speech cooldown window (prevents echo trigger)

    All microphone reads go through audio_stream_lock to prevent contention
    with record_command() and flush_audio_buffer().

    Resilient to transient audio errors: logs the error and continues
    rather than permanently killing wake-word detection.
    """
    global wake_detected, audio_listener_paused
    print("Wake word listener running.")
    while True:
        try:
            with audio_listener_lock:
                paused = audio_listener_paused
            with mode_lock:
                current_mode = robot_mode

            # Don't read from the mic at all if paused or not in robot mode
            if paused or current_mode != MODE_EMOTION:
                time.sleep(0.01)
                continue

            with audio_stream_lock:
                data = audio_stream.read(CHUNK, exception_on_overflow=False)
            pcm = np.frombuffer(data, dtype=np.int16)

            # Skip during audio playback
            with aplay_proc_lock:
                is_playing = current_aplay_proc is not None
            if is_playing:
                continue
            # Skip during post-speech cooldown
            with speech_finished_lock:
                if time.time() - speech_finished_at < WAKE_COOLDOWN_AFTER_SPEECH:
                    continue

            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("Wake word detected!")
                log_wake_word("detected")
                cancel_all_speech()
                with audio_listener_lock:
                    audio_listener_paused = True
                wake_detected = True
        except Exception as e:
            print(f"Audio listener error: {e}")
            time.sleep(0.5)
            continue


audio_thread = threading.Thread(target=audio_listener, daemon=True)
audio_thread.start()
print("Wake word listener started.")


# =============================================================================
#  SECTION 21: LOAD EMOTION MODEL
# =============================================================================

# HSEmotion with EfficientNet-B0, trained on AffectNet.
# The 'enet_b0_8_best_afew' variant was chosen because it supports 8 emotion
# classes and runs efficiently on the Raspberry Pi 5 (~150-300 ms per inference).
print("Loading emotion model...")
emotion_recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')
print("Emotion model ready.")


# =============================================================================
#  SECTION 22: INITIALISE SERVOS & CAMERAS
# =============================================================================

# Set initial servo positions — neutral pose
set_servo_ticks(L_EYELID_CH, int(eyelid_norm_to_ticks(0.35, L_EYELID_OPEN, L_EYELID_CLOSED)))
set_servo_ticks(R_EYELID_CH, int(eyelid_norm_to_ticks(0.35, R_EYELID_OPEN, R_EYELID_CLOSED)))
set_servo_ticks(BEAK_CH, BEAK_CLOSED)
set_servo_ticks(L_WING_CH, int(wing_norm_to_ticks(0.0, L_WING_CLOSED, L_WING_OPEN)))
set_servo_ticks(R_WING_CH, int(wing_norm_to_ticks(0.0, R_WING_CLOSED, R_WING_OPEN)))

neck_now = float(NECK_CEN)
apply_eye_servos()

# Start cameras
cam_left  = setup_cam(0)
cam_right = setup_cam(1)

win_left  = "Left Eye"
win_right = "Right Eye"
cv2.namedWindow(win_left,  cv2.WINDOW_NORMAL)
cv2.namedWindow(win_right, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_left,  640, 480)
cv2.resizeWindow(win_right, 640, 480)


# =============================================================================
#  SECTION 23: ANIMATION HELPERS FOR PARROT MODE
# =============================================================================

def animate_while_waiting(cam_left, cam_right, done_event):
    """Run face tracking, blinking, and wing flutter while waiting for a
    background task (e.g. transcription) to complete.

    Args:
        cam_left, cam_right: Picamera2 instances.
        done_event:          threading.Event that signals the task is finished.
    """
    while not done_event.is_set():
        now = time.time()
        left_result, right_result = capture_and_process_frames(cam_left, cam_right)
        target = choose_active_target(left_result, right_result, now)
        update_tracking(target, apply_tilt_limit=False)
        update_blink(now)
        update_wings(now)

        cv2.imshow(win_left,  left_result["frame"])
        cv2.imshow(win_right, right_result["frame"])
        cv2.waitKey(1)
        time.sleep(0.016)


# =============================================================================
#  SECTION 24: MAIN LOOP
# =============================================================================

parrot_response_text = None
emotion_label  = "neutral"
detected_label = "neutral"
emotion_conf   = 0.0

try:
    # Initialise logging if enabled (inside try so cleanup runs on failure)
    init_logging()

    while True:
        with mode_lock:
            current_mode = robot_mode

        # ─────────────────────────────────────────────────────────────────
        #  PARROT MODE: Wake word detected → record → transcribe → repeat
        # ─────────────────────────────────────────────────────────────────
        if wake_detected:
            wake_detected = False
            with mode_lock:
                robot_mode   = MODE_PARROT
                current_mode = MODE_PARROT
            pixels.fill((255, 255, 255))
            pixels.show()
            print("Switched to parrot mode.")

            # Record user's speech with LED countdown
            flush_audio_buffer(4)
            wav_path = record_command()

            # Start transcription in background: the thread sets the event
            # directly when done, so no extra polling thread is needed.
            with parrot_transcript_lock:
                parrot_transcript       = None
                parrot_transcript_ready = False
            transcribe_done = threading.Event()
            threading.Thread(
                target=run_transcription_thread,
                args=(wav_path, transcribe_done), daemon=True,
            ).start()

            # Animate while transcribing (blink + wings + face tracking)
            pixels.fill((50, 50, 50))
            pixels.show()
            print("Animating while transcribing...")
            animate_while_waiting(cam_left, cam_right, transcribe_done)

            # ── Transcription done — process and speak ────────────────────
            with parrot_transcript_lock:
                transcript = parrot_transcript

            action, response_text = transcript_action(transcript)
            print(f"Action: {action}  Response: {response_text}")
            parrot_response_text = response_text if action == "speak" else None

            # ── Speak repeated text with animation ────────────────────────
            if parrot_response_text:
                parrot_speech_done = threading.Event()

                def parrot_speech_thread():
                    """Speak the repeated text and play an end squawk.
                    Checks speech generation between steps so cancellation
                    takes effect promptly."""
                    my_gen = speech_generation_snapshot()
                    try:
                        speak_with_beak(parrot_response_text, expected_gen=my_gen)
                        if not speech_is_current(my_gen):
                            return
                        end_squawk = get_random_after_squawk()
                        animate_beak_for_wav(end_squawk, expected_gen=my_gen)
                    finally:
                        parrot_speech_done.set()

                threading.Thread(target=parrot_speech_thread, daemon=True).start()
                animate_while_waiting(cam_left, cam_right, parrot_speech_done)

            # ── Return to robot mode ────────────────────────────────────
            with mode_lock:
                robot_mode = MODE_EMOTION
            with speaking_emotion_lock:
                speaking_emotion = None  # safety net in case a speech thread didn't clear it
            with emotion_lock:
                emotion_label = stable_emotion
            update_belly_leds(emotion_label)
            flush_audio_buffer(6)
            with speech_finished_lock:
                speech_finished_at = time.time()
            phrase_last_spoken  = time.time()
            last_spoken_emotion = stable_emotion
            with audio_listener_lock:
                audio_listener_paused = False
            continue

        # ─────────────────────────────────────────────────────────────────
        #  ROBOT MODE: Normal face tracking + emotion detection
        # ─────────────────────────────────────────────────────────────────
        now = time.time()

        # Capture and run face detection on both cameras
        left_result, right_result = capture_and_process_frames(cam_left, cam_right)
        target          = choose_active_target(left_result, right_result, now)
        is_left_active  = target is not None and target["label"] == "L"
        is_right_active = target is not None and target["label"] == "R"

        # ── Face detected: track with eyes and neck ───────────────────────
        if target is not None:
            face_last_seen_time  = now
            no_face_phrase_timer = 0.0
            scan_state = SCAN_IDLE
            scan_next  = time.time() + 5.0

            update_tracking(target, apply_tilt_limit=True)

            # Run emotion detection periodically in background thread
            with emotion_lock:
                busy = emotion_busy
            if not busy and (now - emotion_last_run) >= EMOTION_INTERVAL:
                face_crop = crop_face(target["frame"], target["bbox"],
                                      target["h"], target["w"])
                if face_crop is not None:
                    with emotion_lock:
                        emotion_busy = True
                    emotion_last_run = now
                    threading.Thread(
                        target=run_emotion,
                        args=(face_crop.copy(),), daemon=True,
                    ).start()

        # ── No face: hold position briefly, then scan and drift ─────────
        else:
            face_gone_duration = now - face_last_seen_time

            # Reset emotion to unknown after 3 seconds without a face
            if no_face_phrase_timer != 0.0 and now - no_face_phrase_timer >= 3.0:
                with emotion_lock:
                    stable_emotion       = "unknown"
                    current_emotion      = "unknown"
                    current_emotion_conf = 0.0
                    emotion_history.clear()
                    last_seen_emotion    = "unknown"

            # Hold last tracked position during brief face losses (e.g. blinks
            # covering the camera).  Only start scanning after the face has
            # been absent for FACE_LOST_HOLD_TIME seconds.
            if face_gone_duration >= FACE_LOST_HOLD_TIME:
                # Scanning state machine: pan neck left ↔ right
                if scan_state == SCAN_IDLE:
                    if time.time() >= scan_next:
                        scan_state = scan_direction
                elif scan_state == SCAN_LEFT:
                    neck_now -= SCAN_SPEED * (NECK_CEN - NECK_LEFT)
                    if neck_now <= NECK_LEFT:
                        neck_now       = float(NECK_LEFT)
                        scan_next      = time.time() + SCAN_PAUSE
                        scan_direction = SCAN_RIGHT
                        scan_state     = SCAN_IDLE
                elif scan_state == SCAN_RIGHT:
                    neck_now += SCAN_SPEED * (NECK_RIGHT - NECK_CEN)
                    if neck_now >= NECK_RIGHT:
                        neck_now       = float(NECK_RIGHT)
                        scan_next      = time.time() + SCAN_PAUSE
                        scan_direction = SCAN_LEFT
                        scan_state     = SCAN_IDLE

                # Eyes follow neck direction during scan
                neck_norm  = (neck_now - NECK_CEN) / max(1, NECK_RIGHT - NECK_CEN)
                l_pan_norm += 0.05 * (neck_norm - l_pan_norm)
                r_pan_norm += 0.05 * (neck_norm - r_pan_norm)
                world_tilt_norm *= 0.97

                apply_eye_servos()

        # ── Log tracking data ─────────────────────────────────────────────
        log_tracking(target, now)

        # ── Update belly LEDs to match emotion ────────────────────────────
        # If Freddy is mid-speech, the display emotion is locked to what
        # he's saying.  Otherwise it follows the live stable emotion.
        with speaking_emotion_lock:
            display_override = speaking_emotion
        with emotion_lock:
            detected_label = current_emotion
            emotion_conf   = current_emotion_conf
            if display_override is not None:
                emotion_label = display_override
            else:
                emotion_label = stable_emotion
        update_belly_leds(emotion_label)

        # ── Emotion phrase trigger (squawk + TTS in background) ───────────
        if target is not None and emotion_label not in ("unknown",):
            if now - phrase_last_spoken >= PHRASE_COOLDOWN:
                phrase      = get_emotion_phrase(emotion_label)
                squawk_path = get_random_emotion_squawk(emotion_label)
                print(f"Freddy says: {phrase}")

                with audio_listener_lock:
                    audio_listener_paused = True
                # Lock the display emotion to what Freddy is about to say,
                # so eyes/wings/LEDs stay consistent during speech.
                with speaking_emotion_lock:
                    speaking_emotion = emotion_label

                def emotion_speak_sequence(s, p):
                    """Play emotion squawk then speak phrase, re-enabling
                    the audio listener when done.  Checks speech generation
                    between steps so a wake-word interrupt stops the sequence."""
                    global audio_listener_paused, speech_finished_at, speaking_emotion
                    my_gen = speech_generation_snapshot()
                    try:
                        time.sleep(1.0)  # brief pause before squawk
                        if not speech_is_current(my_gen):
                            return
                        animate_beak_for_wav(s, expected_gen=my_gen)
                        if not speech_is_current(my_gen):
                            return
                        speak_with_beak(p, expected_gen=my_gen)
                    finally:
                        # Release the display emotion lock so detection
                        # takes over again.
                        with speaking_emotion_lock:
                            speaking_emotion = None
                        with speech_finished_lock:
                            speech_finished_at = time.time()
                        # Only unpause listener if still in robot mode —
                        # prevents a cancelled thread from restoring state
                        # after a mode switch.
                        with mode_lock:
                            still_emotion = (robot_mode == MODE_EMOTION)
                        if still_emotion:
                            with audio_listener_lock:
                                audio_listener_paused = False

                threading.Thread(
                    target=emotion_speak_sequence,
                    args=(squawk_path, phrase), daemon=True,
                ).start()
                last_spoken_emotion = emotion_label
                phrase_last_spoken  = now

        # ── No-face phrase trigger ────────────────────────────────────────
        if target is None:
            if no_face_phrase_timer == 0.0:
                no_face_phrase_timer = now
            elif now - no_face_phrase_timer >= NO_FACE_PHRASE_DELAY:
                if now - phrase_last_spoken >= PHRASE_COOLDOWN:
                    phrase = get_no_face_phrase()
                    print(f"No face phrase: {phrase}")

                    with audio_listener_lock:
                        audio_listener_paused = True

                    def no_face_speak(p):
                        """Speak a no-face phrase and re-enable the listener.
                        Checks speech generation so a wake-word interrupt
                        stops the sequence."""
                        global audio_listener_paused, speech_finished_at
                        my_gen = speech_generation_snapshot()
                        try:
                            time.sleep(1.0)
                            if not speech_is_current(my_gen):
                                return
                            speak_with_beak(p, expected_gen=my_gen)
                        finally:
                            with speech_finished_lock:
                                speech_finished_at = time.time()
                            with mode_lock:
                                still_emotion = (robot_mode == MODE_EMOTION)
                            if still_emotion:
                                with audio_listener_lock:
                                    audio_listener_paused = False

                    threading.Thread(
                        target=no_face_speak,
                        args=(phrase,), daemon=True,
                    ).start()
                    phrase_last_spoken = now
                    no_face_phrase_timer = now  # restart delay for next phrase

        # ── Debug overlay ─────────────────────────────────────────────────
        disp_left  = left_result["frame"].copy()
        disp_right = right_result["frame"].copy()
        draw_eye_debug(disp_left,  left_result,  is_active=is_left_active)
        draw_eye_debug(disp_right, right_result, is_active=is_right_active)

        emotion_colour = EMOTION_COLOURS.get(emotion_label, (200, 200, 200))
        target_text   = f"target: {target['label']}" if target else "target: none"
        for img in (disp_left, disp_right):
            cv2.putText(img,
                f"{target_text}  stable: {emotion_label}  detected: {detected_label} ({emotion_conf:.0f}%)",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_colour, 2)

        l_pan_ticks = norm_to_ticks(l_pan_norm, L_PAN_CEN, L_PAN_LEFT, L_PAN_RIGHT)
        r_pan_ticks = norm_to_ticks(r_pan_norm, R_PAN_CEN, R_PAN_LEFT, R_PAN_RIGHT)
        l_tilt_ticks = norm_to_ticks(world_tilt_norm, L_TILT_CEN, L_TILT_UP, L_TILT_DOWN)
        cv2.putText(disp_left,
            f"L pan={int(l_pan_ticks)} lpn={l_pan_norm:.2f} tilt={int(l_tilt_ticks)}",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(disp_right,
            f"R pan={int(r_pan_ticks)} rpn={r_pan_norm:.2f} neck={int(neck_now)}",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(win_left,  disp_left)
        cv2.imshow(win_right, disp_right)

        # ── Blink state machine (emotion-driven) ─────────────────────────
        params = get_emotion_params()
        update_blink(now, params, rest_position=params["e_rest"])

        # ── Wing state machine (emotion-driven) ──────────────────────────
        update_wings(now, params)

        # ── Frame pacing ─────────────────────────────────────────────────
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# =============================================================================
#  SECTION 25: CLEANUP
# =============================================================================
finally:
  # Delete the latest temporary voice recording when the program stops
    try:
        if os.path.exists(OUTPUT_WAV):
          os.remove(OUTPUT_WAV)
          print("Deleted temporary command.wav during cleanup.")
    except Exception as e:
      print(f"Could not delete command.wav during cleanup: {e}")
    # Stop any in-progress audio playback before releasing hardware
    stop_current_playback()

    pixels.fill((0, 0, 0))
    pixels.show()
    cv2.destroyAllWindows()
    face_detection.close()
    pca.deinit()
    cam_left.stop()
    cam_right.stop()
    audio_stream.stop_stream()
    audio_stream.close()
    audio.terminate()
    porcupine.delete()
    close_logs()
