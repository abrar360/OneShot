import pyttsx

engine = pyttsx.init()
voices = engine.getProperty('voices')
rate = engine.getProperty('rate')

#for voice in voices:

engine.setProperty('voice', voices[1].id)

engine.setProperty('rate', rate-50)
engine.say('That object is most likely a banana')
engine.runAndWait()
