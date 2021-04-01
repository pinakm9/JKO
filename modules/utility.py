from time import time

def timer(func):
	"""
	Timing wrapper for a generic function.
	Prints the time spent inside the function to the output.
	"""
	def new_func(*args, **kwargs):
		start = time()
		val = func(*args,**kwargs)
		end = time()
		print('Time taken by {} is {} seconds'.format(func.__name__, end-start))
		return val
	return new_func


def tester(func):
	"""
	Testing wrapper for functions
	"""
	def new_func(*args, **kwargs):
		val = func(*args, **kwargs)
		print('\n' + '#'*80)
		print('Testing {} with args {}'.format(func.__name__, *args))
		print('#'*80)
		print('\nComputed value: {}\n'.format(val))
		print('#'*80 + '\n')
		return val
	return new_func
