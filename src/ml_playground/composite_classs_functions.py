
  

############### Function Composition ###############
# composite_function accepts N number of function as an argument and then compose them
from typing import Callable
import numpy as np

def composite_function(*functions: Callable) -> Callable:
    def composed(x):
        for func in reversed(functions):
        #for func in functions:    
            x = func(x)
        return x
    return composed

# Use Cases 1
def normalize(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / np.std(x)

def log_transform(x: np.ndarray) -> np.ndarray:
    return np.log1p(x)

def round_values(x: np.ndarray) -> np.ndarray:
    return np.round(x, 2)

data = np.array([1, 2, 3, 4, 5])
transform_pipeline = composite_function(round_values, log_transform, normalize)
result = transform_pipeline(data)
print(result)  # Applies normalize, then log_transform, then round_values

# Use Cases 2
import string
def to_lower(text: str) -> str:
    return text.lower()

def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))

def tokenize(text: str) -> list[str]:
    return text.split()

text = "Hello, World! This is a Test."
preprocess = composite_function(tokenize, remove_punctuation, to_lower)
result = preprocess(text)  # Applies to_lower, then remove_punctuation, then tokenize
print(result)


# Applying reduce function in funtools
from functools import reduce
from typing import Callable

def composite_functione(*functions: Callable) -> Callable:
    def compose(f, g):
        return lambda x : f(g(x))
    return reduce(compose, functions, lambda x : x)
# lambda x : x is not necessary

# Note: 
# f(g(x)) - the last element in the functions list is the outermost function.
# g(f(x)) - the first element in the function list is the innermost function.

# or simpley apply lambda 
def composite_functione(*functions: Callable) -> Callable:
    # *functions is a tuple (iterable)
    return reduce(lambda f, g: lambda x: f(g(x)), functions)


"""
functools.reduce(function, iterable[, initializer])
Python's functools module applies a binary function (a function taking two arguments) iteratively to a sequence of elements (iterable), reducing the sequence to a single value.

function: A function that takes two arguments and performs an operation on them.
iterable: An iterable whose elements are processed by the function.
initializer (optional): A starting value for the operation. If provided, it is placed before the first element in the iterable.


functools.reduce()        # if you need one final result 
itertools.accumulate()    # if you need all intermediate results 

# Summing numbers with reduce and lambda
import functools
import itertools

a = [1, 2, 3, 4, 5]
res = functools.reduce(lambda x, y: x + y, a, 0) 

import itertools
res = list(itertools.accumulate(a, lambda x, y: x + y, initial=0)) 

note: the order of function and iterable is different when there are provided as positional arguments
"""


############### Class Composition ###############
"""
Composition: a class Composite can contain an object of another class Component. This type of relationship is known as Has-A Relation.
Inheritance: a Child Class inherit all the properties (is inherited or derived) from the Parent Class. This type of relationship is a.k.a. Is-A Relation. The Child is a specilized version of the Parent.  
Classes that contain objects of other classes are usually referred to as composites, while classes that are used to create more complex types are referred to as components.

At its core, composition involves constructing complex objects by including objects of other classes within them, known as components.
A class that incorporates one or more such objects is referred to as a composite class.
This design pattern allows a composite class to utilize the functionalities of its component classes, without inheriting from them directly. Essentially, the composite class "has a" (or many) component of another class, whereas in class inheritance case, the Child "is a" version of the Parent. isinstance(Child, Parent) returns True.


Function composition:
from typing import Callable

def composite_function(*functions: Callable) -> Callable:
    def composed(x):
        for func in reversed(functions):
        #for func in functions:    
            x = func(x)
        return x
    return composed

"""

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Shape:
    def __init__(self, points):
        self.points = points

triangle = Shape(
    [
        Point(0,0),
        Point(5,5),
        Point(2, 4)

    ]
)

triangle.points


### Defining the Components
class Teacher:
    def __init__(self, name, subject):
        self.name = name
        self.subject = subject
    
    def get_details(self):
        return f"{self.name} teaches {self.subject}."

### Creating the Composite Class
class Department:
    def __init__(self, name):
        self.name = name
        self.teachers = []  # Composition happens here

    def add_teacher(self, teacher):
        self.teachers.append(teacher)
    
    def get_department_details(self):
        details = f"Department: {self.name}\n"
        details += "Teachers:\n"
        for teacher in self.teachers:
            details += f"- {teacher.get_details()}\n"
        return details
    
### Using Composition in Practice    
# Creating teacher instances
teacher1 = Teacher("Alice Smith", "Mathematics")
teacher2 = Teacher("Bob Johnson", "Science")

# Creating a department and adding teachers to it
math_science_department = Department("Math & Science")
math_science_department.add_teacher(teacher1)
math_science_department.add_teacher(teacher2)

# Displaying department details
print(math_science_department.get_department_details())


############### ABC and  @abstractmethod ###############
"""
The abc module formally define the abstract base classes. Abstract base classes exist to be inherited, but never instantiated.
An ABC provides a blueprint for other classes to follow, defining methods that must be implemented by its subclasses. This is useful in scenarios where you want to set a common interface for a group of related classes.

The @abstractmethod decorator is used to indicate that a method is an abstract method. This means that the method does not need to be implemented in the abstract class itself, but must be implemented in any subclass that inherits from the abstract class.
@abstractmethod decorator from the abc module, serve as blueprints for methods that must be implemented by concrete subclasses.



"""

from abc import ABC, abstractmethod

# Here we define a new abstract base class called MediaFile by inheriting from ABC. This class will serve as a template (blue print, design intent) for any subclass.
class MediaFile(ABC):
    def __init__(self, name):
        self.name = name
# @abstractmethod decorator is applied to the pplay() method, indicating that this method is an abstract method. In the body of the method, we typically don't implement any functionality (hence the pass statement) because the implementation should be provided by the subclass. The method defined without any executable code is known as abstract method. The implementation enforcement of the method in all the subclasses guarantees that all subclasses adhere to a common interface, reducing the risk of runtime errors due to unimplemented methods and enhancing the robustness of the software.
    @abstractmethod
    def play(self):
        pass
# Python uses abstract base classes (ABC module) and @abstractmethod decorator to create interfaces. Interfaces serve as contracts that define a set of methods and properties that implementing classes must adhere to. 

# Concrete Classes implementation of the MediaFile abstract class
class AudioFile(MediaFile):
    def play(self):
        return f"Playing audio file: {self.name}"

class VideoFile(MediaFile):
    def play(self):
        return f"Playing video file: {self.name}"
    
# Creating the Composite Class
class MediaPlayer:
    def __init__(self):
        self.playlist = []

    def add_media(self, media_file: MediaFile):
        # note: media_file refers itself for type hint
        # just like recursion - a function calls itself
        self.playlist.append(media_file)

    def play_all(self):
        for media in self.playlist:
            print(media.play())
            # polymorphism - play() method

# Creating instances of media files
audio1 = AudioFile("song1.mp3")
video1 = VideoFile("video1.mp4")

# Creating the media player
player = MediaPlayer()

# Adding media files to the player's playlist
player.add_media(audio1)
player.add_media(video1)

# Playing all media in the playlist
player.play_all()



from abc import ABC, abstractmethod
from typing import Optional, List

class ProjectTask(ABC):
    """Represents a task within a data science project."""

    @abstractmethod
    def get_effort_estimate(self) -> float:
        """Returns the effort estimate to complete the task."""

class DataCollectionTask(ProjectTask):
    """Task related to data collection efforts."""

    def __init__(self, data_sources: int):
        self.data_sources = data_sources

    def get_effort_estimate(self) -> float:
        # Assume each data source requires a fixed amount of effort
        return 2.0 * self.data_sources

class AnalysisTask(ProjectTask):
    """Task for data analysis."""

    def __init__(self, complexity_level: int):
        self.complexity_level = complexity_level

    def get_effort_estimate(self) -> float:
        # Higher complexity increases effort linearly
        return 5.0 * self.complexity_level

class ModelingTask(ProjectTask):
    """Machine Learning modeling task."""

    def __init__(self, number_of_models: int):
        self.number_of_models = number_of_models

    def get_effort_estimate(self) -> float:
        # Assume each model requires a substantial amount of effort
        return 10.0 * self.number_of_models

class DataScienceEmployee:
    """Represents an employee working on data science projects."""

    def __init__(self, name: str, id: int, project_tasks: List[ProjectTask], base_salary: float, bonus: Optional[float] = None):
        self.name = name
        self.id = id
        self.project_tasks = project_tasks
        self.base_salary = base_salary
        self.bonus = bonus

    def compute_compensation(self) -> float:
        """Compute the total compensation including base salary and bonus for task completion."""
        total_effort = sum(task.get_effort_estimate() for task in self.project_tasks)
        compensation = self.base_salary
        if self.bonus is not None:
            compensation += self.bonus * total_effort
        return compensation

def main():
    """Demonstrate the data science project management system."""

    alice_tasks = [DataCollectionTask(data_sources=5), AnalysisTask(complexity_level=3)]
    alice = DataScienceEmployee(name="Alice", id=101, project_tasks=alice_tasks, base_salary=70000, bonus=150)

    bob_tasks = [ModelingTask(number_of_models=2)]
    bob = DataScienceEmployee(name="Bob", id=102, project_tasks=bob_tasks, base_salary=85000, bonus=300)

    print(f"{alice.name} has tasks with a total effort estimate of {sum(task.get_effort_estimate() for task in alice_tasks)} and total compensation of ${alice.compute_compensation()}.")
    print(f"{bob.name} has tasks with a total effort estimate of {sum(task.get_effort_estimate() for task in bob_tasks)} and total compensation of ${bob.compute_compensation()}.")

if __name__ == "__main__":
    main()



# Example Scenario: Predictive Modeling
from typing import List, Callable
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator  # For typing the model

class PreprocessingStrategy: # minimal abstract base classes
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this!")

class EvaluationStrategy: # minimal abstract base classes
    def evaluate(self, model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray) -> float:
        raise NotImplementedError("Subclasses should implement this!")

class StandardScalerPreprocessing(PreprocessingStrategy):
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        scaler = StandardScaler()
        return scaler.fit_transform(data)

class MinMaxScalerPreprocessing(PreprocessingStrategy):
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)
    
class SimpleImputerPreprocessing(PreprocessingStrategy):
    def preprocess(self, data: np.ndarray) -> np.ndarray:  # Renamed 'inputation' to 'preprocess' for consistency
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        return imputer.fit_transform(data)  # Changed to fit_transform for consistency

class PrepocessingChain(List[PreprocessingStrategy]):
     """A sequence of data processing strategies."""
     def preprocess(self, data: np.ndarray, preprocessing_strategies: List[PreprocessingStrategy]) -> np.ndarray:
        result = data
        for strategy in reversed(preprocessing_strategies):
            result = strategy.preprocess(result)
        return result

class RMSEEvaluation(EvaluationStrategy):
    def evaluate(self, model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray) -> float:
        predictions: np.ndarray = model.predict(X_test)
        mse: float = np.mean((predictions - y_test) ** 2)
        return np.sqrt(mse)
    
class PredictiveModel:
    def __init__(
        self,
        model: BaseEstimator,
        preprocessing_strategies: PrepocessingChain,
        evaluation_strategy: EvaluationStrategy
    ) -> None:
        self.model: BaseEstimator = model
        self.preprocessing: PrepocessingChain = preprocessing_strategies
        self.evaluation: EvaluationStrategy = evaluation_strategy


    def train(self, X_train, y_train):
        X_train_processed = self.preprocessing.preprocess(X_train)
        self.model.fit(X_train_processed, y_train)

    def evaluate(self, X_test, y_test):
        X_test_processed = self.preprocessing.preprocess(X_test)
        return self.evaluation.evaluate(self.model, X_test_processed, y_test)

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load and split data
data = fetch_openml(data_id=42165, as_frame=True)
data.data.select_dtypes(include='number')
X_train, X_test, y_train, y_test = train_test_split(data.data.select_dtypes(include='number'), data.target, test_size=0.2, random_state=42)

# Initialize model with strategies
model = PredictiveModel(LinearRegression(), [StandardScalerPreprocessing(), SimpleImputerPreprocessing()], RMSEEvaluation())

# Train and evaluate the model
model.train(X_train, y_train)
rmse = model.evaluate(X_test, y_test)
print(f"RMSE: {rmse}")



# with ABC
# Step 1: Modifying the Abstract Base Class
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

class ModelEvaluator(ABC):
    def __init__(self, X, y, test_size=0.2, random_state=42, metrics=None):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.model = None
        # Set default metrics if none are provided
        if metrics is None:
            metrics = [accuracy_score]  # default to using accuracy if no metrics are specified
        self.metrics = metrics

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def evaluate_model(self):
        predictions = self.predict()
        results = {}
        for metric in self.metrics:
            score = metric(self.y_test, predictions)
            results[metric.__name__] = score
        for name, score in results.items():
            print(f"{name}: {score:.2f}")

# Step 2: Implementing the Abstract Class for Specific Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# Logistic Regression Evaluator
class LogisticRegressionEvaluator(ModelEvaluator):
    def train_model(self):
        self.model = LogisticRegression(max_iter=200)
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        return self.model.predict(self.X_test)

# Decision Tree Evaluator
class DecisionTreeEvaluator(ModelEvaluator):
    def train_model(self):
        self.model = DecisionTreeClassifier()
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        return self.model.predict(self.X_test)
    
# Step 3: Evaluating the Models with Custom Metrics
iris = load_iris()
X, y = iris.data, iris.target

# When we instantiate the evaluators, we can now specify which metrics to use.
metrics = [accuracy_score, precision_score, recall_score]

print("Evaluating Logistic Regression with multiple metrics:")
logistic_evaluator = LogisticRegressionEvaluator(X, y, metrics=metrics)
logistic_evaluator.train_model()
logistic_evaluator.evaluate_model()

print("\nEvaluating Decision Tree with multiple metrics:")
decision_tree_evaluator = DecisionTreeEvaluator(X, y, metrics=metrics)
decision_tree_evaluator.train_model()
decision_tree_evaluator.evaluate_model()



############### Polymorphism ###############
"""
The word "polymorphism" means "many forms", and in programming it refers to methods/functions/operators with the same name that can be executed on many objects or classes.
Polymorphism in Functions. Duck typing enables functions to work with any object regardless of its type. Duck typing is a concept where the type or class of an object is determined by its behavior (i.e., methods and attributes) rather than its explicit inheritance. The name comes from the saying: "If it looks like a duck, swims like a duck, and quacks like a duck, then it probably is a duck."
Compile-time Polymorphism vs Runtime Polymorphism
Polymorphism in Operators. Operator Overloading, + behaves polymorphically, performing addition, concatenation or merging based on the data type. 

Example1: operator +
print(5 + 10)  # Integer addition
print("Hello " + "World!")  # String concatenation
print([1, 2] + [3, 4])  # List concatenation

Example2: len() method
print(len("Hello World!")) # 12
print(len(("apple", "banana", "cherry"))) # 3
print(len({"brand": "Ford",  "model": "Mustang","year": 1964})) # 3

In OOP, polymorphism allows methods in different classes to share the same name but perform distinct tasks. 

"""
# Class Polymorphism example 1 - method area()
class Shape:
    def area(self):
        return "Undefined"

class Rectangle(Shape):
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2

shapes = [Rectangle(2, 3), Circle(5)]
for shape in shapes:
    print(f"Area: {shape.area()}")

# Class Polymorphism example 2 - move()
class Vehicle:
  def __init__(self, brand, model):
    self.brand = brand
    self.model = model

  def move(self):
    print("Move!")

class Car(Vehicle):
  pass

class Boat(Vehicle):
  def move(self):
    print("Sail!")

class Plane(Vehicle):
  def move(self):
    print("Fly!")

car1 = Car("Ford", "Mustang") #Create a Car object
boat1 = Boat("Ibiza", "Touring 20") #Create a Boat object
plane1 = Plane("Boeing", "747") #Create a Plane object

for x in (car1, boat1, plane1):
  print(x.brand)
  print(x.model)
  x.move()


############### Static method vs Staticmethod vs Instance method ###############
class MyClass:
    count = 0

    def __init__(self, value):
        self.instance_variable = value

    # Instance method
    def instance_method(self):
        """This method operates on the instance's data."""
        print(f"Instance variable: {self.instance_variable}")


    @classmethod
    def increment_count_class_method(cls):
        """This method operates on the class's data."""
        cls.count += 1
        return cls.count
    
    @staticmethod
    def static_method(a, b):
        """This method is a utility function within the class."""
        return a + b

# # Create an instance and call the instance method
obj = MyClass("Test instance method")
obj.instance_method()

# # Call the class method using the class name
for i in range(4):
    new_obj = MyClass.increment_count_class_method()
    print(new_obj)

# # Call the static method using the class name
result = MyClass.static_method(5, 7)
print(f"Static method result: {result}")

"""
In Python, methods are functions attached to a given class.

Static method: A staticmethod in Python is a special type of method that is defined inside a class but does not have access to any instance or class data (neither self nor cls parameters, @staticmethod decerator is optional). A staticmethod is used to define a utility function that is related to the class, but does not require access to any instance or class data. A staticmethod is not bound to any specific instance of the class, and can be called on the class itself without the need for an instance. A staticmethod needs no specific parameters. It behaves like a regular function, but is defined inside a class for organizational purposes. Best for utility functions that do not need access to class data. 

Class Method: In Python, classmethod is a type of method is similar to a staticmethod, but it can access and modify the class itself. A class method takes cls as the first parameter for accessing and modifying the class itself (class-level attributes). Best for class-level data or require access to the class itself.  

Instance Method: A method defined within a class, taking 'self' as the first parameter, representing the instance. The most commone method. Best for operations on instances of the class (object).
"""