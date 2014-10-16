import datetime
import sys
import time
"""
includes tools that display time taken and expected finishing time of a loop
being run using printTimeDetails which takes input for i (current loop) n
(final loop) and st(starting time using time.time() before the loop started
"""


def printTimeDetails(i, n, st):
    dt = time.time()-st
    sys.stdout.write(str(round(float(i)/n * 100, 3))
                     + '% complete | Elapsed: '
                     + str(datetime.timedelta(seconds=round(dt)))
                     + ' | ETA: '
                     + str(datetime.timedelta(seconds=(round(dt*n/float(i))
                                                       - round(dt))))
                     + '          \r')
    sys.stdout.flush()


def loadingBar(i, n):
    # will use 40 '=' signs to display progress through a loop
    percent = float(i) / n * 100
    numbEquals = int(percent*.4)
    if(i != (n-1)):
        sys.stdout.write('[' + numbEquals * '=' + ' ' * (40-numbEquals)
                         + ']' + str(round(percent, 4)) + '% complete'
                         + '                    \r')
    else:
        sys.stdout.write('['+40*'='+'] Done'+'              \r')
