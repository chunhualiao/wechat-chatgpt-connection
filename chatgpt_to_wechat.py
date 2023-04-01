# Origional code https://github.com/marticztn/WeChat-ChatGPT-Automation
# modified by: C. Liao
#  step 1. get the window
#   step 2: taking screenshots
#   step 3: extracting text
import openai
import time
import cv2
import pyautogui as auto
import numpy
import pyperclip
from PIL import Image
# from AppKit import NSPasteboard, NSStringPboardType
import clipboard
from colorthief import ColorThief
import json

import pygetwindow as gw
import sys
import numpy as np
import tempfile
#import pytesseract # OCR quality is bad for Chinese.
#import easyocr  # OCR quality is bad for mixed Enlighs and Chinese, also slow on CPU only machine
import logging
from paddleocr import PaddleOCR
from dotenv import load_dotenv
import os

import time

# Configure logging level for PaddleOCR
#logging.getLogger("paddleocr").setLevel(logging.ERROR)
# Set up the logging level
# logging.basicConfig(level=logging.ERROR)
# Locate the logger used by PaddleOCR
paddle_logger = logging.getLogger('ppocr')

# Set the logging level to ERROR (suppress DEBUG and WARNING messages)
paddle_logger.setLevel(logging.ERROR)

prev_image_array = np.array([],dtype=float)
# load the contents of the .env file into os.environ
load_dotenv()

# get the value of the API_KEY environment variable
api_key = os.environ.get('API_KEY')

if api_key is None:
    print("OpenAI's API key not found in environment variables")
    sys.exit(1)
    
MODEL = "gpt-3.5-turbo"
API_KEY = api_key
IMG_NAME = '1.png'
openai.api_key = API_KEY

# global questions
questions = {}

# TODO: best use "greenshot" to find the coordinates of the chat window's input box
# the input message box's start position
# using this as the anchor point!!
inx = 1035
iny = 1062

# shift to hit middle of Copy menu item
plusx = 41
plusy = 14

# The greenshot program can be used to find the coordinates of the chat window.
# MUST update the region coordinates for your own computer!!
# the chat message display region's coordinates
left = 1315
top = 64
width = 1851
height = 670

# where to grab the actual message content from the screenshot
# the output message box's left bottom corner
mx = 1401
my = 642


# msgs is a list of dictionaries.
# the first dictionary is the initial message from the assistant
msgs = [
    {"role": "system", "content": "你是一个很幽默的助手"}
]

# No longer used
# Analyze screenshot
# continuously take screenshots of a region on the screen and
# analyze the color of the background and the content.
def analyzeScreenshot():
    # background color RBG values
    # paint program will find out RGB values of the background color
    # used as a reference to determine if the screen content has changed
    prev_avg_color = numpy.array([245, 245, 245])
    while True:
        # x,y, width, height
        auto.screenshot(IMG_NAME, region=(left, top, width, height))
        img = cv2.imread(IMG_NAME)  # image file is loaded as a numpy array

        # obtain the dominant color of the screenshot
        thief = ColorThief(IMG_NAME)
        dom_color = thief.get_color(quality=1)

        avg_color_per_row = numpy.average(img, axis=0)
        avg_color = numpy.average(avg_color_per_row, axis=0)
        color_arr = numpy.array(img)
        fin_arr = color_arr.tolist()

        # print(dom_color)
        # print(auto.position())
        # check if the background color has changed
        # RGB 20,20,20 is black, 24,24,24 is gray
        if not (numpy.array_equal(avg_color, prev_avg_color)) and \
                dom_color != (20, 20, 20) and \
                dom_color != (24, 24, 24) and \
                [24, 24, 24] not in fin_arr[99]:
            content: str = copyMessage()

            bot_prefix = '@chatgpt'
            if content.startswith(bot_prefix):
                response = getAnswer(content.removeprefix(bot_prefix))
                print(response, end='\n')
                pyperclip.copy(response)
                sendMessage(response)
            else:
                print('Skipping message without matching prefix')
                # time.sleep(0.2)  # only sleep when no request is sent
            # pyperclip.copy(response)
            # sendMessage(response)

        prev_avg_color = avg_color


# Copy message content from the chat window, top left corner at (a,b) position
# all new messages will show up in the top left corner of the screen eventually.
def copyMessage() -> str:
    global msgs
    # move the mouse cursor to the message box and copy message
    # simulating the process of copying the message using the mouse
    # ??? this is the message window's position on my computer.
    a = mx  # 10
    b = my  # 10
    auto.moveTo(a, b)
    auto.rightClick()
    # shift to the right and down : hit middle of copy!!
    auto.moveTo(a+plusx, b+plusy)
    auto.click()  # we should click on the "copy" option now
    auto.moveTo(a, b)

    # time.sleep(0.2) # we don't need this may be

    # pb = NSPasteboard.generalPasteboard()
    pb = clipboard.paste()  # bug: this does not work as expected!!

    # pbstring: str = pb.stringForType_(NSStringPboardType)
    pbstring: str = pb
    msgs.append({"role": "user", "content": pbstring})

    return pbstring

# move the mouse cursor to the message box and send message using pasting
# getgreenshot.org software: printscreen to find the coordinates.


def sendMessage(msg: str):
    # this position is for my computer's window layout only.
    # TODO: must update for your own computer
    auto.moveTo(inx, iny)
    auto.click()
    # auto.hotkey('command', 'v')  # pyautogui
    auto.hotkey('ctrl', 'v')
    auto.press('enter')

# Calling ChatGPT API
# add answer to the msgs list, with a size limit of 25


def getAnswer(msg: str) -> str:
    # global variable msgs, accessible anywhere in the program
    # msgs is a list of dictionaries!
    global msgs
    print('New message detected: ' + msg)

    # append the user's message info as a dictionary to the msgs list
    msgs.append({"role": "user", "content": msg})
    print(msgs)

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=msgs,
        # max_tokens = 100,
        temperature=0.8
    )['choices'][0]['message']['content'].strip()
    # strip() removes the leading and trailing spaces

    # append the response info. as a dictionary into the msgs list
    msgs.append({"role": "assistant", "content": response})

    list_as_str = json.dumps(msgs)
    tokens = list_as_str.split()
    token_count = len(tokens)
    # This checks if the length of the msgs list has exceeded 25 elements,
    # and if so, removes the second and third elements of the list.
    # This ensures that the msgs list does not grow too large and consume too much memory.
    if len(msgs) >= 10 or token_count > 2000:
        msgs.pop(1)  # list index id starts from 0
        msgs.pop(2)

    return response

# the main program
# analyzeScreenshot()


# Set the Tesseract OCR path to the installation path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path accordingly
# Update this path accordingly
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# capture the new questions from the chat window
def capture_chat_text(window_title):
    global prev_image_array
    global questions
    result =[]      
    windows = gw.getWindowsWithTitle(window_title)
    if len(windows) <= 0:
        print("No window with title '{}' found.".format(window_title))
        sys.exit(1)

    window = windows[0]

    # window = gw.getWindowsWithTitle(window_title)[0]

    if window.visible:
        # Get the window position and size
        x, y, width, height = window.left, window.top, window.width, window.height

        # Capture the window screenshot
        # the returned screenshot variable's type is PIL.Image.Image
        # pyautogui
        screenshot = auto.screenshot(region=(x, y, width, height))

        current_image_array = np.array(screenshot)
        
        if np.array_equal(prev_image_array, current_image_array):
            print("The images are the same. No new questions.")
            time.sleep(5) # must sleep for a while, otherwise the program will run too fast
            return result
        else:
            print("The images are different. Detecting new questions...")        
        prev_image_array = current_image_array
        
        # Initialize PaddleOCR with mixed languages (English and Simplified Chinese)
        ocr = PaddleOCR(lang='ch')

        # Perform OCR on the screenshot
        chat_text = ocr.ocr( current_image_array)


        for line in chat_text[0]:
                line= line[-1][0]
                # check if the line contains a prefix string of "@chatgpt"
                if line.startswith('@chatgpt'):
                    # get the message
                    question = line.split('@chatgpt')[1]   
                    # trim the leading and trailing spaces
                    question = question.strip()   
                    if question not in questions:
                        questions[question] = 1
                        # print(question)                                                    
                        # add the new question into a list of results
                        result.append(question)
                        
        return result                                                       
# Each inner list contains information about the detected text line,
# including the bounding box coordinates
# (in the format [x1, y1, x2, y2, x3, y3, x4, y4]), confidence score, and the recognized text.
# The recognized text can be accessed using the last element of the inner list (e.g., line[-1]).
#         [
#     [[x1, y1, x2, y2, x3, y3, x4, y4], confidence, 'Text line 1'],
#     [[x1, y1, x2, y2, x3, y3, x4, y4], confidence, 'Text line 2'],
#     ...
# ]

        #-------------- pytesseract OCR: bad for Chinese recogniztion. 
        # Convert the screenshot to a grayscale image
        #img_gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2GRAY)
        # Extract text from the image
        # Extract text using Tesseract OCR with Simplified Chinese language
        #text = pytesseract.image_to_string(img_gray, lang='chi_sim') # no control over temp path to write text file
        # Create temporary file for output
        # Print the extracted text
        #print(text)

        #-------------- using easyocr: bad for English mixed within Chinese. 
        # Convert the screenshot to grayscale and resize it        #
        # img_gray = screenshot.convert('L')
        # img_gray = img_gray.resize((img_gray.width*3, img_gray.height*3))
        # img_gray = np.array(img_gray)
        
        # # Perform OCR on the screenshot
        # reader = easyocr.Reader(['ch_sim'])
        # result = reader.readtext(img_gray)

        # result = [
        #     ((10, 10, 200, 50), '你好', 0.9),
        #     ((10, 60, 300, 100), '世界', 0.95),
        # ]       
    else:
        print("Chat window named '{}' not found or not visible.".format(window_title))
        return None


if __name__ == '__main__':
    
    start_time = time.time()
    
    if sys.version_info >= (3, 11):
        sys.exit("This script only supports Python versions earlier than 3.11, due to the OSCR package used.")

    # Get all visible windows
    windows = gw.getAllWindows()

    # Print the title of each window
    print('---------------------------------')
    print("All visible windows' titles are:")
    for window in windows:
        if window.visible:            
            print(window.title)
    
    # Replace this with the actual window title of the chat program
    chat_window_title = 'chatgpt test'

    while True:
        new_questions = capture_chat_text(chat_window_title)
        if new_questions:
            print('---------------------------------')
            for line in new_questions:                
                answer = getAnswer(line)
                print(answer)
                # copy the answer to the clipboard
                pyperclip.copy(answer)
                # paste the answer to the chat window
                sendMessage(answer)            
        else:
            print("No new questions extracted from the Chat window.")
            #time.sleep(1)  # Adjust the monitoring interval as needed
        elapsed_time = time.time() - start_time
        print(f">>>>>>>  Elapsed time: {elapsed_time:.2f} seconds")
        # if 
        if elapsed_time > 3600:
            print(f">>>>>>>  Elapsed time exceeding limit, exiting...")
            break

            
