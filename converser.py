from dotenv import load_dotenv
import speech_recognition as sr
from playsound import playsound
from gtts import gTTS
import openai
import spacy 
import os 

load_dotenv()


api_keys = os.getenv('API_KEY')
spacy.prefer_gpu()
NLP = spacy.load('en_core_web_sm')
openai.api_key = api_keys

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source=source)
        print("listening.......")
        try:
            text = recognizer.recognize_google_cloud(audio_data=audio)
            print("you said "+text)
            return text
        except sr.UnknownValueError:
            text2 = "error i didn't get that"
            print(text2)
            return text2
        except sr.RequestError:
            text3 = "error having issue with the connection"
            print(text3)
            return text3



def process_text(text):
    doc = NLP(text=text)
    tokens = [(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
               token.shape_, token.is_alpha, token.is_stop) for token in doc]
    dataTypes = []
    for token in tokens:
        print(spacy.explain(token[2]))
        dataTypes.append(spacy.explain(token[2]))
    if any(spacy.explain(token[2]) == "verb" for token in tokens):
        return "command", dataTypes
    elif doc[-1].text == "?":
        return "question", dataTypes
    else: return "statment", data


def generate_answer(question):
    response = openai.completions.create()


def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    playsound('response.mp3')
    os.remove('response.mp3')


#  "screw the nut on the board"
text = "put a nut anywhere you see a black color"

typeText, dataType = process_text(text)


# python3 converser.py
