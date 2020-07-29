# import future print function to override it
from __future__ import print_function
# use pretty print at the core
from pprint import *
import sys
import inspect
import numpy as np

"""
Author: Warren Lemmon
This file magically indents your console output by the combined indentation of the source code and all calling functions' cumulative indentation:
Example:
hi from call stack a
	hi from call stack a within an if block
	hi from call stack a->b
		hi from call stack a->b within an if block
			hi from call stack a->b:if->c
			
This file vents a decorator @printIndentDecorator.
Use it like this:
---------------------
from __future__ import print_function
from print_indentation_utils import printIndentDecorator
@printIndentDecorator
def myfunc():
	print('hi')
--------------------
Notice: you cannot use print 'hi' as normal python if you import this file. However, all files that do not import this file will still operate normally.
	
Details of operation: 
The decorator inspects the source code of the decorated method and temporarily increments the global variable cumulative indent.
The print method is overwritten to apply the cumulative indent. When a print statement is called, it prepends the cumulative indent to the statement.
Decorating your methods will (hopefully) still allow your code to be executed from an interpreter or from other callers unaware of this utility.
The utility is also quite computationally efficient. It only adds minor overhead, as it only performs object inspections one per method.

Additionally, this file vends a class wrapper to decorate all methods in a class:
Use it like this:
--------------------------------------
@for_all_methods(printIndentDecorator)
class MyClass(Object):
	def __init__(self):
		print('hi')
-------------------------------------


"""

# parameters

# maxPrintIndentDepth sets a cutoff as to how many indentations the util will allow.
maxPrintIndentDepth=None
#maxPrintIndentDepth =3

# line width for numpy arrays, matrices, etc.
np.set_printoptions(linewidth =170)

# switch to enable or disable the util if it is causing issues
usePrintIndentationUtils = True
#usePrintIndentationUtils = False




	
# do not touch these. They are part of the utility.
global cumulativeIndent, sourceLineCache
cumulativeIndent = None
sourceLineCache = {}

def for_all_methods(decorator):
	def decorate(cls):
		for attr in cls.__dict__: # there's propably a better way to do this
			if callable(getattr(cls, attr)):
				setattr(cls, attr, decorator(getattr(cls, attr)))
		return cls
	return decorate

	
# this decorates each stack frame to provide the cumulative indentation since the call to setInitIndent()
def printIndentDecorator(f):	
	def inner_dec(*args, **kwargs):
		global cumulativeIndent, sourceLineCache
		if cumulativeIndent is None:
			setInitIndent()
		perform = usePrintIndentationUtils and (maxPrintIndentDepth is None or cumulativeIndent <= maxPrintIndentDepth)
		if perform:
			#sys.stdout.write(str(inspect.getsourcelines(inspect.currentframe().f_code)))
			# get the frame above the current one; this is the decorated frame.
			frame = inspect.currentframe().f_back
			# look up in cache for efficiency
			if frame.f_code in sourceLineCache:
				(lines, firstlineno, firstlineindent) = sourceLineCache[frame.f_code]
			else:
				# outer-most frame not working; do this instead
				if frame.f_back is None:
					fname = inspect.getfile(frame)
					# get all source lines for file
					try:
						with open(fname) as file:
							lines = file.readlines()
						firstlineno = 1
						firstlineindent = 0
					except IOError as e:	
						# if this is running from commandline, we cannot indent first frame
						perform = False
				else:
					(lines, firstlineno) = inspect.getsourcelines(frame.f_code)
					firstlineindent = getIndent(lines[0])
				# if we just fetched lines, firstline, and indent cache them
				if perform:
					sourceLineCache[frame.f_code] = (lines, firstlineno, firstlineindent)
			# if still good, calculate indent and increment indent
			if perform:
				
				currentLineNo = frame.f_lineno
				#sys.stdout.write('firstlineindent'+str(firstlineindent)+"\n\nlines"+str(lines)+"\n\ncurrentLineNo:"+str(currentLineNo)+"\n\nfirstlineno:"+str(firstlineno)+"\n")
				#sys.stdout.write('currentLineNo - firstlineno'+str(currentLineNo - firstlineno)+"\nlen(lines)"+str(len(lines))+"\n")
				
				lineIndexOfInterest = currentLineNo - firstlineno
				#sys.stdout.write('\nfirstlineno'+str(firstlineno))
				#sys.stdout.write('\nlineIndexOfInterest'+str(lineIndexOfInterest)+'\n')
				#sys.stdout.write('\nlen(lines)'+str(lineIndexOfInterest)+"\n")
				
				line = lines[lineIndexOfInterest]
				#sys.stdout.write('line:'+line)
				lineindent = getIndent(line)
				additionalIndent = lineindent - firstlineindent
				#sys.stdout.write('lineindent:'+str(lineindent)+", firstlineindent:"+str(firstlineindent)+"\n")
				
				cumulativeIndent += additionalIndent
				#sys.stdout.write("cumulativeIndent:"+str(cumulativeIndent)+"\n")
		# run original function
		res = f(*args, **kwargs)
		# put indent back to normal
		if perform:
			cumulativeIndent -= additionalIndent
			#sys.stdout.write(str(cumulativeIndent)+"\n")
		return res
	return inner_dec
	
	
def setInitIndent():
	global cumulativeIndent
	# this method will be called with one indent greater than our desired base indentation, 
	# so we set the indent to -1. next, the printIndentDecorator will decrement the indent further.
	# when it comes time to print at the first level debug block, 
	# the cumulative indent will be set to zero.
	cumulativeIndent = -1

def getIndent(line):
	numSpacesUsedForIndent = 4
	#assert len(line.lstrip(" ")) == len(line)
	#sys.stdout.write(line+str("\n"))
	#sys.stdout.write(line.lstrip("\t")+"\n")
	return max(len(line) - len(line.lstrip("\t")), (len(line) - len(line.lstrip(" ")))/numSpacesUsedForIndent)

# this replaces the default print method with an indented, prettyprint version
class IndentPrint(object):
	def __init__(self):
		self.indentDepth = 4
		self.pprints = PrettyPrinter(indent=1, width=80, depth=None, stream=None)
		self.indentChar = ' ' * self.indentDepth
	
		self._stdout = sys.stdout
	@printIndentDecorator
	def write(self,*args):
		global maxPrintIndentDepth, cumulativeIndent
		if maxPrintIndentDepth is not None and cumulativeIndent > maxPrintIndentDepth:
			return
		
		indentString = self.indentChar * cumulativeIndent
		string = self.getString(args, indentString)
		self._stdout.write(string+"\n")
		return string
	__call__ = write
	def getString(self, args, indentString):
		string = indentString
		for arg in args:
			if isinstance(arg, basestring):
				string += arg
			else:
				string += self.pprints.pformat(arg).replace("\n", "\n"+indentString)
			string += ' '
		return string
	def getLineNo(self):
		pass
		return lineno

print = IndentPrint()

@printIndentDecorator
def t11():
	print('indented 2')
	if True:
		print('indented 3')

@printIndentDecorator
def t1():
	print('indented 1')
	t11()

@printIndentDecorator
def t2():
	t11()

if __name__ == '__main__':
	print('indented 0')
	if True:
		print('indented 1')
	t1()
	t2()
	t2()
	t2()