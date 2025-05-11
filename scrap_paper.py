import serial
import time

ser = serial.Serial(
    port='COM4',        # Change this to your serial port (e.g., '/dev/ttyUSB0' on Linux/Mac)
    timeout=0, 
    rtscts=True     
)

time.sleep(3)

#sleep complete

try:
    # Data to send
   
    message = "off\n" 
    # Send the message
    ser.write(bytes(message,'utf-8'))

except serial.SerialException as e:
    print(f"Serial error: {e}")
finally:
    # Close the serial port
    ser.close()