import PySimpleGUI as sg 
import os
import numpy as np 
import cv2
import requests
import matplotlib.pyplot as plt 
from PIL import Image, ImageDraw, ImageFont
import globalvariables
import io
import time
from supporting_functions import *

analyze_url = globalvariables.MSA_END_POINT
headers = {'Prediction-Key': globalvariables.MSA_PREDICTION_KEY,
           'Content-Type': 'application/octet-stream'}

layout = [
    
    [sg.Text('Compliance Detection')],
    
    [sg.Slider(orientation ='horizontal', key='--CONFIDENCE--', range=(1,100))],
    
    [
        sg.Button('Start Detection'),
        sg.Button('Stop Detection')
    ],
    
    [
        sg.Text('Camera Feed'),
        sg.Image(key='--CAMERA--')
    ]

]

window = sg.Window('MSA', layout)

cap = cv2.VideoCapture(0)
recording = False
detection = False

cam_id = parse_args()


with Vimba.get_instance():
    with get_camera(cam_id) as cam:
        while True:
            event, values = window.read(timeout = 20)

            recording = True
            
            confidence = int(values['--CONFIDENCE--'])

            if event == 'Exit' or event == sg.WIN_CLOSED:
                break
            
            elif event == 'Start Detection':
                detection = True
                
            elif event == 'Stop Detection':
                detection = False

            if detection:
                fmts = cam.get_pixel_formats ()
                fmts = intersect_pixel_formats(fmts , OPENCV_PIXEL_FORMATS)
                if fmts:
                    cam.set_pixel_format(fmts[0])
                    fmts = fmts[0]
                else:
                    print("No Intersection with the format")
                now_frame = cam.get_frame()
                
                now_frame.convert_pixel_format(fmts)
                
                frame = cv2.cvtColor(now_frame.as_opencv_image(), cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame,  mode='RGB')

                imgbytes = cv2.imencode('.png', now_frame.as_opencv_image())[1].tobytes()
                # window['--CAMERA--'].update(data=imgbytes)

                draw = ImageDraw.Draw(image)

                response = requests.post(analyze_url, headers=headers, data=imgbytes)

                color = 'red'

                for i in response.json()['predictions']:
                    if i['probability'] * 100 >= confidence:
                        detec = i['tagName']
                        confidence_level = i['probability']
                        
                        message = '{} \nprob: {}'.format(detec, confidence_level)

                        left = i['boundingBox']['left']*image.size[0]
                        top = i['boundingBox']['top']*image.size[1]
                        width = i['boundingBox']['width']*image.size[0]
                        height = i['boundingBox']['height']*image.size[1]

                        draw.line([(left, top), (left + width, top)], fill=color, width=5)
                        draw.line([(left + width, top), (left + width, top + height)], fill=color, width=5)
                        draw.line([(left + width, top + height), (left, top + height)], fill=color, width=5)
                        draw.line([(left, top + height), (left, top)], fill=color, width=5)
                        
                        font = ImageFont.truetype('arial.ttf', 10)
                        right = left + width + 5
                        draw.text((right, top), message, (255, 0, 0), font = font)

                buf = io.BytesIO()
                image.save(buf, format='PNG')
                byte_im = buf.getvalue()
                window['--CAMERA--'].update(data = byte_im)
                # time.sleep(5)
                
            if recording and not detection:
                fmts = cam.get_pixel_formats ()
                fmts = intersect_pixel_formats(fmts , OPENCV_PIXEL_FORMATS)
                if fmts:
                    cam.set_pixel_format(fmts[0])
                    fmts = fmts[0]
                else:
                    print("No Intersection with the format")
                now_frame = cam.get_frame()
                now_frame.convert_pixel_format(fmts)
                imgbytes = cv2.imencode('.png', now_frame.as_opencv_image())[1].tobytes()
                window['--CAMERA--'].update(data = imgbytes)

        window.close()

print(1)