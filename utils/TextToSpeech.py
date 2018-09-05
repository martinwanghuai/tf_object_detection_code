'''
Created on 4 Sep 2018

@author: martinwang
'''
# import pyttsx3
# f = open("/Users/martinwang/eclipse-workspace/tf_object_detection_code/tts.txt","r")
# line = f.readline()
# engine = pyttsx3.init()
# while line:
#     line = f.readline()
#     print(line, end = '')
#     engine.say(line)
# engine.runAndWait()
# f.close()    

import pyttsx3
engine = pyttsx3.init()

rate = engine.getProperty('rate')
engine.setProperty('rate', rate+50)
volume = engine.getProperty('volume')
engine.setProperty('volume', volume-0.25)


engine.say('Given user is in initial page')
engine.say('Its awesome.')

voices = engine.getProperty('voices')
for voice in voices:
  engine.setProperty('voice', voice.id)
  engine.say('The quick brown fox jumped over the lazy dog.')


engine.runAndWait()