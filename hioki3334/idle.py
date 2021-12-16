import time
from hioki3334 import hioki3334

dev = hioki3334()
dev.integrate_stop()
dev.integrate_reset()
dev.integrate_start()
history = []
while True:
  time.sleep(1)
  wh, t = dev.measure_wh_time()
  history.append((wh, t))

  wh0, t0 = history[0]
  wh1, t1 = history[-1]
  if t0 < t1:
    avg_watt = (wh1 - wh0) / ((t1 - t0) / 3600)
    print(f'Average watt for last {t1 - t0} sec is {avg_watt:.6f}. (wh0, t0, wh1, t1 = {wh0}, {t0}, {wh1}, {t1})')

  if len(history) > 60:
    history.pop(0)
