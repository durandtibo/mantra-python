
import time
import sys

class ProgressBar:

	def __init__(self, maxSteps, msg="", length=50, refresh=1000):
		self.maxSteps = maxSteps
		self.msg = msg
		self.length = length
		self.refresh = refresh
		self.current = 0
		self.startTime = None
		self.currentTime = None
		self.lastTime = None


	def get_time_millis(self):
		""" Compute the current time in ms"""
		return int(round(time.time() * 1000))


	def start(self):
		self.startTime = self.get_time_millis()
		self.currentTime = self.get_time_millis()
		self.current = 0
		self.force_show(self.startTime)


	def step(self, numSteps=None):
		if self.startTime is None:
			self.start() 
		if numSteps is None:
			self.current += 1
		else: 
			self.current += numSteps
		if self.current > self.maxSteps:
			self.maxSteps = self.current
		self.show()


	def show(self):
		self.currentTime = self.get_time_millis()
		if self.currentTime - self.lastTime > self.refresh or self.refresh == 0:
			self.force_show(self.currentTime)


	def stop(self):
		self.force_show(self.get_time_millis())
		sys.stdout.write('\n')
		sys.stdout.flush()


	def progress(self):
		if self.maxSteps is 0:
			return 0
		else:
			return int(round(self.length * self.current / float(self.maxSteps)))


	def eta(self, elapsed):
		if self.maxSteps == 0 or self.current == 0:
			return "?"
		else:
			etaMillis = (float(elapsed) / float(self.current)) * (self.maxSteps - self.current)
			return time.strftime("%H:%M:%S", time.gmtime(etaMillis / 1000))


	def percentage(self):
		if self.maxSteps is 0:
			return '? %'
		else:
			percent = int(self.current / float(self.maxSteps) * 100)
			if percent < 10:
				return '  {}%'.format(percent)
			elif percent < 100:
				return ' {}%'.format(percent)
			else:
				return '{}%'.format(percent)


	def force_show(self, currentTime):
		self.lastTime = self.get_time_millis()
		progress = self.progress()
		percent = self.percentage()
		bar = '=' * progress + '.' * (self.length - progress)
		elapsedTime = self.currentTime - self.startTime
		elapsedString  = time.strftime("%H:%M:%S", time.gmtime(elapsedTime / 1000))
		etaString = self.eta(elapsedTime)
		sys.stdout.write('\r%s %s [%s]  %d/%d (%s / ETA %s)' % (self.msg, percent, bar, self.current, self.maxSteps, elapsedString, etaString))
		sys.stdout.flush()


	def __str__(self):
		return 'ProgressBar [maxSteps=%s, length=%s, current=%s]' % (self.maxSteps, self.length, self.current)




def debug():
	print('debug progress bar')
	pb = ProgressBar(100, msg='pb')
	#pb.start()
	
	for i in range(100):
		pb.step()
		time.sleep(0.1)

	pb.stop()

	timeString  = time.strftime("%H:%M:%S", time.gmtime(112364))
	print(timeString)


if __name__ == "__main__": debug()




