import time
import board
import busio
import adafruit_mlx90640
import numpy as np
import  cv2
import tflite_runtime.interpreter as tflite


TFLITE_FILE_PATH = '/home/pi/Desktop/testred/model.tflite'
interpreter =  tflite.Interpreter(TFLITE_FILE_PATH)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_ind={
    0:'ambiente',
    1:'objeto',
    2:'persona'    
}


i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)


mlx = adafruit_mlx90640.MLX90640(i2c)
print("MLX addr detected on I2C")
print([hex(i) for i in mlx.serial_number])

mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ

imagen = 255*np.ones((600,800,3),dtype=np.uint8)


preds=0

frame = [0] * 768

while True:

    mlx.getFrame(frame)

    
    print("Read 2 frames in %0.2f s" % (time.monotonic() - stamp))
    for h in range(24):
        for w in range(32):
            t = frame[h * 32 + w]
            if t < 20:
                    rigby =(102,0,0)
                    
            elif t < 23:
                    rigby =(153,0,0)
            elif t < 25:
                    rigby =(204,0,0)
            elif t < 27:
                    rigby =(255,0,0)
            elif t < 29:
                    rigby =(255,255,204)
            elif t < 31:
                    rigby =(153,255,255)
            elif t < 33:
                    rigby =(51,255,255)
            elif t < 35:
                    rigby =(102,178,255)
            elif t < 37:
                    rigby =(51,153,255)
            elif t < 39:
                    rigby =(0,76,153)
            elif t >39:
                    rigby=(0,25,51) 
                    
            cv2.rectangle(imagen,(w*10,h*10),(w*10+10,h*10+10),rigby,-1) #imagen(donde se vizualiza el rectangulo),punto inicial, punto final,BGR,grosor de linea(1 solo una lineap -1 todo el cuadro).   
                
            print(t)
        print()
        
    img=cv2.rectangle(imagen,(w*10,h*10),(w*10+10,h*10+10),rigby,-1)
    img=img[0:240,0:320]
    
        
    if t>=36:
                
       img=cv2.resize(img,(224,224))
       input_shape = input_details[0]['shape']
       input_tensor= np.array(np.expand_dims(img,0))
       input_index = interpreter.get_input_details()[0]["index"]
       interpreter.set_tensor(input_index, input_tensor)
       interpreter.invoke()
       output_details = interpreter.get_output_details()
       output_data = interpreter.get_tensor(output_details[0]['index'])
       pred = np.squeeze(output_data)
       highest_pred_loc = np.argmax(pred)
        
       preds = class_ind[highest_pred_loc]
       
       print(preds)
    
    if cv2.waitKey(1) & 0xFF==ord('q') :
       break
    

cv2.destroyAllWindows()