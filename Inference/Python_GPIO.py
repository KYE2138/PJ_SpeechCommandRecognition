import RPi.GPIO as GPIO
import time

# to use Raspberry Pi board pin numbers
GPIO.setmode(GPIO.BOARD)
LED_PIN = 16
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.HIGH)
print("on!")
time.sleep( 5 )
GPIO.cleanup() # 把這段程式碼放在 finally 區域，確保程式中止時能夠執行並清掉GPIO的設定！

GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)
print("off！")
time.sleep( 5 )
GPIO.cleanup()

FAN_PIN = 18
GPIO.setmode(GPIO.BOARD)
GPIO.setup(FAN_PIN, GPIO.OUT)

p = GPIO.PWM(FAN_PIN, 25000)
p.start(0)
try:
    while 1:
        for dc in range(0, 101, 5):
            p.ChangeDutyCycle(dc)
            time.sleep(1)
        time.sleep(10)
        for dc in range(100, -1, -5):
            p.ChangeDutyCycle(dc)
            time.sleep(1)
except KeyboardInterrupt:
    p.stop()
    GPIO.cleanup()
    pass
