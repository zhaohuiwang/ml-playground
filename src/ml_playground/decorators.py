
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

###### @typing.runtime_checkable ######
"""
The @typing.runtime_checkable decorator marks a Protocol class as capable of being used with runtime checks like isinstance() and issubclass(). Without this decorator, Protocol classes are primarily for static type checking and cannot be used in runtime checks.

Protocol:
1. Internal protocols, such as the iterator, context manager, and descriptor protocols. These protocols consist of special methods that make up a given protocol. For example, the .__iter__() and .__next__() methods define the iterator protocol.
2. The second protocol term specifies the methods and attributes that a class must implement to be considered of a given type. This feature allows you to enforce a relationship between types or classes without the burden of inheritance. This relationship is known as structural subtyping or static duck typing.
For example, built-in container types such as lists, tuples, strings, dictionaries, and sets all support iteration.
Dynamic: Python interpreter checks an object’s type when the code runs and can change during the variable's lifetime.
x =4
x = "x is a string now"
Duck typing allows for flexible and dynamic code. With duck typing, you can use different and unrelated objects in a given context if those objects have the expected methods and attributes. You don't have to ensure that the objects share a common parent type through inheritance.


"""
from typing import Protocol, runtime_checkable

@runtime_checkable
class Greetable(Protocol):
    """A protocol for objects that can be greeted."""
    def greet(self, name: str) -> str:
        ...

class Person:
    def greet(self, name: str) -> str:
        return f"Hello, {name}!"

class Robot:
    def greet(self, name: str) -> str:
        return f"Greetings, human {name}."

class NonGreetable:
    def say_hi(self) -> str:
        return "Hi!"

# Create instances
person_instance = Person()
robot_instance = Robot()
non_greetable_instance = NonGreetable()

# Perform runtime checks using isinstance()
print(f"Is person_instance Greetable? {isinstance(person_instance, Greetable)}")
print(f"Is robot_instance Greetable? {isinstance(robot_instance, Greetable)}")
print(f"Is non_greetable_instance Greetable? {isinstance(non_greetable_instance, Greetable)}")

# Example of using the protocol
def introduce(obj: Greetable, target_name: str):
    print(obj.greet(target_name))

if isinstance(person_instance, Greetable):
    introduce(person_instance, "Alice")

if isinstance(robot_instance, Greetable):
    introduce(robot_instance, "Bob")


class Duck:
    def quack(self):
        return "The duck is quacking!"
class Person:
    def quack(self):
        return "The person is imitating a duck quacking!"

def make_it_quack(duck: Duck) -> str:
    return duck.quack()

print(make_it_quack(Duck()))
print(make_it_quack(Person()))
# error: Argument 1 to "make_it_quack" has incompatible type "Person"; expected "Duck"  [arg-type]

# One way you can fix this issue is to use inheritance
class QuackingThing:
    def quack(self):
        raise NotImplementedError(
            "Subclasses must implement this method"
        )

class Duck(QuackingThing):
    def quack(self):
        return "The duck is quacking!"
class Person(QuackingThing):
    def quack(self):
        return "The person is imitating a duck quacking!"
    
def make_it_quack(duck: QuackingThing) -> str:
    return duck.quack()

print(make_it_quack(Duck()))
print(make_it_quack(Person()))

""" 
Two ways to decide whether two objects are compatible as types:
Nominal subtyping is strictly based on inheritance. A class that inherits from a parent class is a subtype of its parent(how the built-in isinstance() works).
Structural subtyping is based on the internal structure of classes. Two classes with the same methods and attributes are structural subtypes of one another.
"""

class Dog:
    def __init__(self, name):
        self.name = name
    def eat(self):
        print(f"{self.name} is eating.")
    def drink(self):
        print(f"{self.name} is drinking.")
    def make_sound(self):
        print(f"{self.name} is barking.")

class Cat:
    def __init__(self, name):
        self.name = name
    def eat(self):
        print(f"{self.name} is eating.")
    def drink(self):
        print(f"{self.name} is drinking.")
    def make_sound(self):
        print(f"{self.name} is meowing.")
# We could define an Animal class and Bog(Animal), Cat(Animal) subclass (Nominal subtyping). But here Dog and Cat don’t have a strict inheritance relationship. They’re completely decoupled and independent classes. But they have the same internal structure, including methods and attributes(Structural subtype), thus can be used in a duck typing context.
for animal in [Cat("Tom"), Dog("Pluto")]:
    animal.eat()
    animal.drink()
    animal.make_sound()
# However, up to this point, you don’t have a formal way to indicate that Cat and Dog are subtypes. To formalize this subtype relationship, you can use a protocol.
# protocols allow you to specify the expected methods and attributes that a class should have to support a given feature without requiring explicit inheritance. So, protocols are explicit sets of methods and attributes.








### Protocols vs Abstract Base Classes ###

from abc import ABC, abstractmethod
from math import pi

class Shape(ABC):
    """
    The @abstractmethod decorator in Python, imported from the abc module, is used to define abstract methods within abstract base classes (ABCs). It signifies that a method must be implemented by any concrete (non-abstract) subclass that inherits from the abstract class. This ensures that subclasses adhere to a specific interface or contract defined by the abstract methods. 
    """
    @abstractmethod
    def get_area(self) -> float:
        pass

    @abstractmethod
    def get_perimeter(self) -> float:
        pass

class Circle(Shape):
    def __init__(self, radius) -> None:
        self.radius = radius

    def get_area(self) -> float:
        return pi * self.radius**2

    def get_perimeter(self) -> float:
        return 2 * pi * self.radius

class Square(Shape):
    def __init__(self, side) -> None:
        self.side = side

    def get_area(self) -> float:
        return self.side**2

    def get_perimeter(self) -> float:
        return 4 * self.side
    

from typing import Protocol
from math import pi

# Add @runtime_checkable decorator to mark a protocol class as a runtime protocol so that you can use it with isinstance() and issubclass().
# @runtime_checkable
class Shape(Protocol):
    def get_area(self) -> float:
        ...

    def get_perimeter(self) -> float:
        ...
# We define a class called Shape by inheriting from typing.Protocol. Shape implements two methods - get_area() and get_perimeter() which deine the Shape protocol itself. Note that protocol methods don't have a body, which you typically indicate with the ellipsis (...) syntax.

class Circle:
    def __init__(self, radius) -> None:
        self.radius = radius

    def get_area(self) -> float:
        return pi * self.radius**2

    def get_perimeter(self) -> float:
        return 2 * pi * self.radius

class Square:
    def __init__(self, side) -> None:
        self.side = side

    def get_area(self) -> float:
        return self.side**2

    def get_perimeter(self) -> float:
        return 4 * self.side
# We define two classes, Circle and SSquare. These classes implement the Shape protocol because they have the same two methods - get_area() and get_perimeter(). 
def print_shape_info(shape: Shape):
    print(f"Area: {shape.get_area()}")
    print(f"Perimeter: {shape.get_perimeter()}")
# We can use objects of either class as arguments to the print_shape_info() function, which takes an Shape object as an argument.
circle = Circle(10)
square = Square(5)
print_shape_info(circle)
print_shape_info(square)

# In short, the main difference between an abstract base class and a protocol is that the former works through a formal inheritance relationship, while the latter doesn’t need this relationship. 
