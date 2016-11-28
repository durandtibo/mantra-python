
class LabeledObject:

	def __init__(self, pattern=None, label=None):
		self.pattern = pattern
		self.label = label

	def __str__(self):
		return 'LabeledObject [pattern=%s, label=%s]' % (self.pattern, self.label)
