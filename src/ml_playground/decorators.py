
"""
A wrapper is a function or method that modifies or extends the behavior of another function without changing its actual code.

How Does a Wrapper Work?
A wrapper is essentially a higher-order function, which means it takes a function as an argument and returns a new function that usually adds some extra logic before or after calling the original function. Wrappers often interact with the function through the use of *args and **kwargs to forward arguments to the wrapped function.

A very common use case of wrappers in Python is decorators.

"""
# Simple Wrapper Example

def wrapper(func):
    def wrapped(*args, **kwargs):
        print("Before calling the function.")
        result = func(*args, **kwargs)
        print("After calling the function.")
        return result
    return wrapped

# Applying the wrapper to a function - decorator
@wrapper
def say_hello(name):
    print(f"Hello, {name}!")

say_hello("Alice")

"""
@staticmethod
Purpose: Used to define a method that doesn't require access to instance-specific data (no self parameter). It belongs to the class but doesn't depend on instance state.

@classmethod is a decorator that defines a method that operates on the class itself (cls) rather than an instance (self)

"""
class MathOperations:
    count = 0

    @classmethod
    def increment_count(cls):
        cls.count += 1
    
    @staticmethod
    def add(x, y):
        return x + y

print(MathOperations.add(3, 5))  # Output: 8

test = MathOperations()
test.count          # Output:  5 

for i in range(5):
    MathOperations.increment_count()
    print(MathOperations.count)

# This class is mutable. The count attribute keeps incrementing over iteration.

test = MathOperations()
test.count          # Output:  5


class Person:
    population = 0

    def __init__(self, name):
        self.name = name
        Person.population += 1

    @classmethod
    def get_population(cls):
        return cls.population

person1 = Person("Alice")
person2 = Person("Bob")
print(Person.get_population())  # Output: 2

""""
@property                       __get__()
Purpose: Used to define a method as a getter for an attribute, allowing you to access it as if it were an attribute (without calling it as a method).

@<property_name>.setter         __set__()
Purpose: Allows you to define a setter method for a property, enabling modification of a "read-only" property.
@<property_name>.deleter        __delete__()

"""
class Rectangle:
    def __init__(self, width, height):
        self._width = width
        self._height = height

    @property
    def area(self):
        return self._width * self._height
    
    # Note: the property_name is area for this case

    @area.setter
    def ajust_area(self, value):
        self._height = value / self._width

rect = Rectangle(4, 5)
print(rect.area)  # Output: 20

rect.area = 40  # Sets a new area by adjusting height
print(rect._height)  # Output: 10


""""
@functools.wraps
Purpose: Used in the implementation of custom decorators to preserve the original function’s metadata (such as name, docstring, etc.) when it’s wrapped by the decorator.

@functools.lru_cache 
Purpose: Caches the results of expensive function calls to optimize repeated calls with the same arguments. This is useful for memoization.
It implements a Least Recently Used (LRU) cache, meaning it discards the least recently used items when the cache reaches its size limit.

Memoization: The decorator stores the function's input arguments and their corresponding output in a cache. If the function is called again with the same arguments, it returns the cached result instead of re-executing the function.
LRU Cache: The cache has a finite size (controlled by the maxsize parameter). When the cache is full, the least recently used items are discarded to make room for new ones.
Thread Safety: The cache is thread-safe, meaning it can be safely used in multi-threaded applications without additional synchronization.
Hashable Arguments: The function's arguments must be hashable (e.g., integers, strings, tuples of hashable objects) for the cache to work.

"""

from functools import wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Before calling the function")
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    """This function greets a person."""
    return f"Hello, {name}!"

print(greet.__name__)  # Output: greet
print(greet.__doc__)   # Output: This function greets a person.


from functools import lru_cache

# Without caching
def fibonacci_no_cache(n):
    if n <= 1:
        return n
    return fibonacci_no_cache(n - 1) + fibonacci_no_cache(n - 2)

# With caching
@lru_cache(maxsize=128)
def fibonacci_with_cache(n):
    if n <= 1:
        return n
    return fibonacci_with_cache(n - 1) + fibonacci_with_cache(n - 2)

# Test
import time

start = time.time()
print(fibonacci_no_cache(40))  # Slow due to redundant calculations
print(f"Without cache: {time.time() - start:.4f} seconds")

start = time.time()
print(fibonacci_with_cache(40))  # Fast due to caching
print(f"With cache: {time.time() - start:.4f} seconds")

# Check cache info
# Shows hits, misses, maxsize, and current size
print(fibonacci_with_cache.cache_info())  


"""
@timer (Custom Decorator Example)
Purpose: A custom decorator to measure the execution time of a function.

"""
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(2)

slow_function()  # Output: slow_function took 2.0001 seconds

@timer
def test_fibonacci_with_cache():
    print(fibonacci_with_cache(40))
test_fibonacci_with_cache()

@timer
def test_fibonacci_no_cache():
    fibonacci_no_cache(40)
test_fibonacci_no_cache()




"""
@abstractmethod (from abc module)
Purpose: Used to define an abstract method in an abstract base class (ABC). It forces subclasses to implement this method.

"""
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        return "Woof!"

dog = Dog()
print(dog.make_sound())  # Output: Woof!



"""
@singleton (Custom Decorator Example)
Purpose: Ensures that a class has only one instance (i.e., the Singleton pattern).

"""
def singleton(cls):
    instances = {}
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return wrapper

@singleton
class Database:
    def __init__(self, name):
        self.name = name

db1 = Database("MyDB")
db2 = Database("AnotherDB")

print(db1 is db2)  # Output: True (db1 and db2 are the same instance)


"""
@contextmanager (from contextlib module)
Purpose: Used to create context managers that allow you to define setup and teardown code using the with statement.

"""
# Simple example
from contextlib import contextmanager

@contextmanager
def file_writer(name):
	try:
		f = open(name, 'w')
		yield f
	finally:
		f.close()

with file_writer("example.txt") as file:
	file.write("hello world")
	
# Alternatively, define context manager in a class
class FileWriter(object):
    def __init__(self, filename):
        self.file_name = filename

    @contextmanager
    def open_file(self):
        try:
            file = open(self.file_name, 'w')
            yield file
        finally:
            file.close()

mf = FileWriter("example.txt")

with mf.open_file() as file:
    file.write("hello world")





# same output can be obtained by the following 
with open("test.txt", "w") as f:
    f.write("Hello, world!")
"""
The yield statement is the key part of the context manager function.

It temporarily pauses the function execution and hands control back to the with block where the context manager is used.

The file object is returned to the with statement, allowing it to be used as a resource inside the block.

After the block finishes, the function will resume execution right after the yield statement (to close the file).
"""




class FileWriter(object):
    def __init__(self, filename):
        self.file_name = filename

    @contextmanager
    def open_file(self):
        try:
            file = open(self.file_name, 'w')
            yield file
        finally:
            file.close()

mf = FileWriter("example.txt")
with mf as file:
    file.write("hello world")

# or just simply
from contextlib import contextmanager

@contextmanager
def file_writer(name):
	try:
		f = open(name, 'w')
		yield f
	finally:
		f.close()

with file_writer("example.txt") as file:
	file.write("hello world")
	
###### @override ######

# The syntax

from typing import override

class Parent:
    def greet(self) -> str:
        return "Hello from Parent!"

class Child(Parent):
    @override
    def greet(self) -> str:
        return "Hello from Child!"


obj = Child()
obj.greet()
 
"""
Available from Python 3.12 onwards. Its purpose is to explicitly indicate that a method in a subclass is intended to override a method inherited from a parent class.

The @override decorator is primarily for static type checking and is optional at runtime. Python's method resolution order (MRO) will still correctly handle method overriding even without the decorator.

The @override decorator helps catch common errors when overriding methods in Python. These errors include:

Method Not Found (Method Names are different)
Return Type Mismatch
Signature Mismatch (Different Parameters between parent and child methods)

"""