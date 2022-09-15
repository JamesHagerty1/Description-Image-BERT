import time

num = 1
while True:
    time.sleep(5)
    num += 1
    f = open('test.txt', 'w')
    f.write(str(num))
    f.close()