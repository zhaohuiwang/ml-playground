

# print(*objects, sep=' ', end='\n', file=None, flush=False)
print('1', '2', '3', sep='-', end='');
print('World!')

a = 5
b = 3
a // b
a % b

a = float(input("Enter th first number:"))
b = float(input("Enter th second number:"))


df = pd.read_csv(filepath_or_buffer, sep=',', header='infer',)
df.to_csv("output.csv", index=False)
df = pd.read_json('filename.json'); df.to_json('filename.json')
df = pd.read_xml('filename.xml'); pd.to_xml()

dfs = pd.read_excel("data.xlsx", sheet_name=["Sheet1", "Sheet2"])
for sheet, data in dfs.items():
    print(f"Sheet name: {sheet}")
    print(data.head())
df.to_excel("output.xlsx", index=False)

df = pd.read_parquet("data.parquet", columns=["name", "age"], engine="pyarrow")
df.to_parquet("output.parquet")


df.head() # Display Top Rows: 
df.tail() # Display Bottom Rows: 
df.dtypes # Display Data Types: 
type(object_name)   # 
df.describe() # Summary Statistics: 
df.info() # Display Index, Columns, and Data:


df.isnull().sum() # Check for Missing Values
df.notnull().all().all() 
# check missing value at cell, column and table levels

df.fillna(value) # Fill Missing Values
df.dropna() # Drop Missing Values
df.rename(columns={'old_name': 'new_name'}) # Rename Columns
df.drop(columns=['column_name']) # Drop Columns
df.replace(np.nan, 0)

import pandas as pd
import numpy as np
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Alice'],
    'Age': [25, np.nan, 30, np.nan, 25]
}
df = pd.DataFrame(data)
df['Name'].fillna(df['Name'].mode()[0], inplace=True) # DataFrame.mode() returns a series
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['column'].interpolate(method='linear')
# method = 'linear', 'polynomial', 'nearest', ...
df['column'].ffill()
df['column'].bfill()


df['column'].apply(lambda x: function(x)) # Apply Function
df.groupby('column').agg({'column': 'sum'}) # Group By and Aggregate
df.pivot_table(index='column1', values='column2', aggfunc='mean') # Pivot Tables
pd.merge(df1, df2, on='column') # Merge DataFrames 
df1.merge(df2, how='inner', on=None, left_on=None, right_on=None)
# how = 'inner', 'outer', 'left', 'right' 
df.pivot(index='id_col', columns='year',values='col')
pd.wide_to_long(df, ['col'], i='id_col', j='year')


pd.concat([df1, df2]) # Concatenate DataFrames
pd.concat(objs, *, axis=0, join='outer')


df['column'].hist()  # Histogram
df.boxplot(column=['column1', 'column2'])  # Boxplot
df.plot.scatter(x='col1', y='col2')  # Scatter Plot 
df.plot.line()  # Line Plot 
df['column'].value_counts().plot.bar()  # Bar Chart


df.corr()  # Correlation Matrix
df.cov()  # Covariance Matrix
df['column'].value_counts()  # Value Counts
df['column'].unique()  # Unique Values in Column
df['column'].nunique()  # Number of Unique Values

df['column'].astype('type')  # Convert Data Types
df['column'].astype('category')
df['column'].str.lower()  # String Operations
pd.to_datetime(df['column'])  # Datetime Conversion
df.set_index('column')  # Setting Index

df.set_index(pd.to_datetime(df['date']))  # Set Datetime Index
df.resample('M').mean()  # Resampling Data
df.rolling(window=5).mean()  # Rolling Window Operations

df['column'].astype('type')  # Convert Data Types
df['column'].str.lower()  # String Operations
pd.to_datetime(df['column'])  # Datetime Conversion
df.set_index('column')  # Setting Index


from pandas_profiling import ProfileReport
ProfileReport(df)

import seaborn as sns; sns.pairplot(df)
sns.heatmap(df.corr(), annot=True)

df.query('column > value')  # Query Function
df[df['column'].isin([value1, value2])]  # Filtering with isin

df.memory_usage(deep=True)  # Return the memory usage of each column in bytes.
 

df.drop_duplicates()
df.duplicated()


df.groupby('col').transform(lambda x: x - x.mean())
df.groupby('col').agg(['mean','sum'])



df[df['column'].str.contains('substring')]
df['column'].str.split(' ', expand=True)
df['column'].str.extract(r'(regex)')


(df['column'] - df['column'].mean()) / df['column'].std()
(df['column'] - df['column'].min()) / (df['column'].max() - df['column'].min())


df['column'] = df['column'].astype('category') # convert column to categorical
df['column'].cat.set_categories(['cat1', 'cat2'], ordered=True)  # ordered categories
# Series.cat.set_categories(*args, **kwargs)


df.reset_index(drop=True)
df.set_index(['col1', 'col2'])  # set multiple indexes

# append and extend methods for list
list_a = [1, 2, 3]
list_b = ['a', 'b']
list_a.append(4)
list_a.append(list_b)
print(list_a)

list_a.extend(list_b)
print(list_a)


content = []
# content attribute is not considered "falsy" (e.g., it's not None, an empty string, or an empty list)
if content:
    print(True)
else:
    print(False)

"""
Easydict
EasyDict allows to access dict values as attributes (works recursively). A Javascript-like properties dot notation for python dicts.

OmegaConf is a YAML based hierarchical configuration system, with support for merging configurations from multiple sources (files, CLI argument, environment variables) providing a consistent API regardless of how the configuration was created.

Python tempfile module is a part of the standard library. Its primary purpose is to manage temporary filesystem resources that are needed during the execution of a program and are automatically cleaned up when no longer required. 


The tarfile module in Python is a standard library module used for reading and writing tar archives, including those that are compressed with gzip (.tar.gz) bz2 (.tar.bz2) and lzma compression. .

"""

from easydict import EasyDict as edict
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

import tempfile     # built-in module
import yaml


cfg = OmegaConf.create(
   {
       "plans": {
           "A": "plan A",
           "B": "plan B",
       },
       "selected_plan": "A",
       "plan": "${plans[${selected_plan}]}",
   }
)
cfg.plan 
'plan A'
cfg.selected_plan = "B"
cfg.plan 
'plan B'

# Usecase 1
# Portions of the config (cfg) are saved with models. If this was run from the command line with hydra, the config getting passed in is not JSON serializable, which throws an error when saving. Dumping to a temp file and reloading is a workaround.

if isinstance(cfg, DictConfig):
# omegaconf.dictconfig.DictConfig is a class  of  hierarchical configuration system that support YAML or dictionaries
    with tempfile.NamedTemporaryFile() as temp:
        OmegaConf.save(config=cfg, f=temp.name)
        # OmegaConf.save() Serializing/save OmegaConf objects in YAML
        temp.seek(0) 
        #  seek() method sets the file's cursor at a specified position in the current file
        cfg = EasyDict(yaml.safe_load(temp))

# Opens a new tar archive named my_archive.tar.gz in write mode with gzip compression ("w:gz") > Adds the file file1.txt to the archive. > Adds the directory directory_to_archive and its contents recursively to the archive.
# The with statement ensures the archive is properly closed after writing.
try:
    with tarfile.open("my_archive.tar.gz", "w:gz") as tar:
        if os.path.exists("file1.txt"):
            tar.add("file1.txt")
        else:
            print("file1.txt not found")
        if os.path.exists("directory_to_archive"):
            tar.add("directory_to_archive")
        else:
            print("directory_to_archive not found")
except Exception as e:
    print(f"Error creating archive: {e}")
# Opens the existing tar archive my_archive.tar.gz in read mode with gzip compression ("r:gz") > Extracts all contents of the archive to the destination_directory folder. If the directory doesn’t exist, it will be created.
# The with statement ensures the archive is properly closed after extraction.
try:
    with tarfile.open("my_archive.tar.gz", "r:gz") as tar:
        tar.extractall("destination_directory")
        print("Extraction successful")
except Exception as e:
    print(f"Error extracting archive: {e}")


############### asynchronous function a.k.a. coroutine ###############

import asyncio

# async def instead of just def to define an asynchronous function. It marks a function as a coroutine, which can be paused and resumed, allowing other code to run during waiting periods (e.g., waiting for a response from a server).
async def task1():
    # The await keyword is used inside an async function to pause execution until the awaited task completes. Only async functions can use await.
    await asyncio.sleep(2)
    print("Task 1 done")

async def task2():
    await asyncio.sleep(1)
    print("Task 2 done")

async def main():
    # Run tasks concurrently
    await asyncio.gather(task1(), task2())


# Always use asyncio.run() for top-level execution or integrate with an event loop.
asyncio.run(main())

# A module
#if __name__ == "__main__":
#    asyncio.run(main())
""""
output:
Task 2 done
Task 1 done
Here, task2 finishes first (1-second delay) while task1 runs concurrently (2-second delay), showing how asyncio handles tasks efficiently.

In Python, async def is used to define an asynchronous function (also called a coroutine) that can run concurrently with other tasks using the asyncio framework. It enables non-blocking code execution, which is useful for I/O-bound operations like network requests, file operations, or database queries, allowing your program to handle multiple tasks efficiently without waiting for each to complete.

Common use cases:
I/O-Bound tasks: HTTP requests, database queries, or file operations.
Web Services:Frameworks like FastAPI or aiohttp use async def for handling multiple client requests concurrently.
Real-Time Applications: Chat applications, streaming or other systems requiring low-latency response.

Coroutine: A function defined with async def creates a coroutine, which doesn't run immediately when called but returns a coroutine object.
Await: The await keyword is used inside an async function to pause execution until the awaited task completes. Only async functions can use await.
Event Loop: The asyncio event loop manages the execution of coroutines, scheduling them to run concurrently.
Concurrency, Not Parallelism: Async functions enable concurrency (tasks taking turns) but not true parallelism (multiple tasks running simultaneously, as with threads or multiprocessing).

"""


############### callback function ###############
# Example 1
def greet(name):
     print(f"Hello, {name}!")


def process_user_input(callback):
     name = input("Enter your name: ")
     callback(name)


process_user_input(greet)
# Enter your name: zhaohui wang
# Hello, zhaohui wang!


# Example 2
def print_result(result):
    """A simple callback function to print the result."""
    print(f"The processed result is: {result}")

def log_result(result):
    """Another callback function to log the result."""
    print(f"Logging result: {result}")

def process_data(data, callback):
    """Processes data and calls a callback function."""
    processed_result = data * 2
    callback(processed_result)


# Using the print_result callback
process_data(5, print_result)

# Using the log_result callback
process_data(10, log_result)

"""
https://realpython.com/ref/glossary/callback/
In Python, a callback is a function that you pass as an argument to another function. The receiving function can then call the callback at a later point, often as a response to an event or after completing a task.

Callbacks are a powerful way to customize the behavior of functions and are commonly used in asynchronous programming, event handling, and GUI applications.

Keras has a built-in, high level callback system - keras.callbacks.Callback class that allow you to customize the behavior of your Keras model during training, evaluation, or prediction. 

PyTorch Lightning provides a robust and extensible callback system - lightning.pytorch.callbacks.Callback, that allows users to inject custom logic at various stages of the training process.


Google ADK callbacks allow you to observe, customize, and even control the agent's behavior at specific, predefined points without modifying the core ADK framework code (Observe & debug, Customize & control, Implement guardrails, Manage state and integrate & enhance).

Before Agent callback and After Agent callback
Before Model callback and After Model callback
Before Tool callback and After Tool callback

The Callback Mechanism: Interception and Control
When the ADK framework encounters a point where a callback can run (e.g., just before calling the LLM), it checks if you provided a corresponding callback function for that agent. If you did, the framework executes your function.

Callback arguments - CallbackContext or ToolContext (Context objects) contains vital information about the current state of the agent's execution, including the invocation details, session state, and potentially references to services like artifacts or memory. You use these context objects to understand the situation and interact with the framework.

Controlling the Flow through the return value from callbacks:
when `return None` (Allow Default Behavior)
when `return <Specific Object>` (Override Default Behavior)

"""
from typing_extensions import override

class MyClass:
    def __init__(self):
        self._data = [1, 2, 3]

    @property
    @override
    def data(self):
        return self._data

my_instance = MyClass()
# Correct way to iterate:
for item in my_instance.data:
    print(item)
    

class MyClass:
    def __init__(self):
        self._data = [1, 2, 3]

    @property
    @override
    def data(self):
        return self._data

    def __iter__(self):
        return iter(self._data)  #  Return an iterator from the underlying iterable.

my_instance = MyClass()
for item in my_instance:  # Now you can iterate directly over the instance.
    print(item)




############### Error Handeling ###############
# raise ValueError vs print
def process_positive_number(num):
    if num <= 0:
        raise ValueError("Input must be a positive number.")
    # When raise ValueError is executed, it immediately interrupts the normal flow of the program. Unless the ValueError is caught and handled by a try...except block, the program will terminate and display a traceback, indicating where the error occurred.
    print("continue after raise ValueError")

try:
    process_positive_number(-5)
except ValueError as e:
    print(f"Error: {e}")

process_positive_number(-5)

process_positive_number(5)



def process_positive_number(num):
    if num <= 0:
        print("continue after raise ValueError")
    # Program execution continues normally after a print() statement, unless other code explicitly causes a halt.
    print("Continuing ...")

process_positive_number(-5)

process_positive_number(5)


############### Environment variables ###############
"""
When the environment variables are saved in .env file, like
DATABASE_URL='https://www.example.com'
API_KEY=your-api-key

And you run a script as a module
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Define the path to the .env file
env_file_path = Path(__file__).parent.parent.parent / ".env"

# Load environment variables from the specified .env file
load_dotenv(dotenv_path=env_file_path)

# Now you can access them using os.getenv()
db_url = os.getenv("DATABASE_URL")
api_key = os.getenv("API_KEY")

print(f"Database URL: {db_url}")
print(f"API Key: {api_key}")


############### Command-line interfaces (CLIs) ###############
"""
sys.argv: A built-in Python list in the sys module that captures command-line arguments as strings. It's the simplest, most basic way to access CLI inputs but requires manual parsing.

argparse: A standard library module for parsing command-line arguments. It provides a robust, flexible framework for defining and validating arguments, generating help messages, and handling complex CLIs.

Click: A third-party library (pip install click) that uses decorators to create concise, user-friendly CLIs with features like nested commands and automatic help generation.

Typer: Similar to Click but leverages type hints, making it more modern and concise for typed Python code. (pip install typer)

Hydra: A third-party framework (pip install hydra-core) designed for configuring complex applications, particularly machine learning experiments. It generates CLIs as a side effect, with support for hierarchical configuration via YAML files.


docopt: Simpler than argparse/Click for small CLIs, but less structured than Hydra for config management. (pip install docopt)

Fire: Easiest for exposing existing code as a CLI, but less predictable than Click or argparse for structured interfaces. (pip install fire)    
"""

# sys.argv Use case: Simple script needing basic argument access.
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: script.py <name> <age>")
        sys.exit(1)
    
    name = sys.argv[1]
    age = int(sys.argv[2])  # Manual type conversion
    print(f"Name: {name}, Age: {age}")

if __name__ == "__main__":
    main()
# Run: python script.py Alice 30
# Output: Name: Alice, Age: 30
# Pros: No dependencies, simple for small scripts.
# Cons: No built-in validation, error handling, or help messages.


# argparse Use case: CLI with options, flags, and help messages.
import argparse

def main():
    parser = argparse.ArgumentParser(description="Greet a user.")
    parser.add_argument("name", help="User's name")
    parser.add_argument("--age", type=int, default=25, help="User's age")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Processing for {args.name} with age {args.age}")
    print(f"Name: {args.name}, Age: {args.age}")

if __name__ == "__main__":
    main()

# Run: python script.py Alice --age 30 --verbose
# Output:
# Processing for Alice with age 30
# Name: Alice, Age: 30
# Run with help: python script.py --help
# Output: Shows auto-generated help message.
# Pros: Built-in, supports types, defaults, and help.
# Cons: Verbose setup for complex CLIs.


# Click Use case: Modern CLI with minimal boilerplate and complex features.
import click

@click.command()
@click.argument("name")
@click.option("--age", type=int, default=25, help="User's age")
@click.option("--verbose", is_flag=True, help="Enable verbose output")
def greet(name, age, verbose):
    if verbose:
        click.echo(f"Processing for {name} with age {age}")
    click.echo(f"Name: {name}, Age: {age}")

if __name__ == "__main__":
    greet()

# Run: python script.py Alice --age 30 --verbose
# Output:
# Processing for Alice with age 30
# Name: Alice, Age: 30
# Run with help: python script.py --help
# Output: Auto-generated help message.
# Pros: Clean syntax, supports complex CLIs, great for user-facing tools.
# Cons: Requires external dependency.

# Hydra Use case: Managing complex configurations, especially for research or ML.

# Typer Description: A modern CLI framework built on top of Click, using type hints for automatic argument parsing. Ideal for Python 3.6+ projects leveraging type annotations.
import typer

app = typer.Typer()

@app.command()
def greet(name: str, age: int = 25, verbose: bool = False):
    if verbose:
        typer.echo(f"Processing for {name} with age {age}")
    typer.echo(f"Name: {name}, Age: {age}")

if __name__ == "__main__":
    app()

# Run: python script.py Alice --age 30 --verbose
# Output:
# Processing for Alice with age 30
# Name: Alice, Age: 30
# Pros: Type-hint driven, clean syntax, great for modern Python projects.
# Cons: Requires external dependency (pip install typer).

    
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(f"Name: {cfg.user.name}, Age: {cfg.user.age}")
    if cfg.verbose:
        print(f"Verbose mode: Processing for {cfg.user.name}")

if __name__ == "__main__":
    main()
'''
# conf/config.yaml
user:
  name: Alice
  age: 25
verbose: false
'''
# Run: python script.py user.name=Bob user.age=30 verbose=true
# Output:
# Name: Bob, Age: 30
# Verbose mode: Processing for Bob

# Pros: Powerful for config management, supports YAML, command-line overrides, and hierarchical configs.
# Cons: Steep learning curve, overkill for simple CLIs.






############### Control Flow Constructs ###############
### if-else Statements on predictable conditions
if condition:
    # Do something
else:
    print("Message")

# a ternary operator for a one-liner
print("Message") if not condition else do_something()

"""
In Python, the `if not variable:` construct is used to check for the "falsiness" of a variable. This means the code block following the if not statement will execute if the variable evaluates to a boolean False. 
How Python evaluates "falsiness":
Python's built-in types have inherent "truthy" or "falsy" values when evaluated in a boolean context. Specifically, the following are considered "falsy":
- None
- False
- Numeric zero values: 0, 0.0, 0j
- Empty sequences: '' (empty string), [] (empty list), () (empty tuple)
- Empty mappings: {} (empty dictionary)
- Empty sets: set()

The `if not variable:` construct is generally preferred for checking emptiness as it is concise and leverages Python's built-in truthiness rules.

For explicitly checking None, `if variable is None:` is the correct and most Pythonic approach.
"""

my_string = ""
my_list = []
my_number = 0

if not my_string:
    print("my_string is empty")
if not my_list:
    print("my_list is empty")
if not my_number:
    print("my_number is zero or falsy")


my_data = None
# or my_data = ""
# or my_data = []

if my_data is None or not my_data:
    print("my_data is None or empty")

# checking for variable existance()NameError
try:
    if my_undefined_variable:
        pass # This line will not be reached if the variable is undefined
except NameError as e:
    print(f"Error: {e}")


# Non-disruptive vs deruption

### try-except Statements for error handeling

### if condition raise
# usually halts the normal flow of the program unless it is caught by an except block.
age = "25"
if not isinstance(age, int):
    raise ValueError("Age should be an integer.")



############### enum (short for enumeration) ###############
# The enum module, introduced in Python 3.4, provides support for creating enumerations using the Enum class. Each member of an enumeration is a unique, named constant, often used to represent a fixed set of options or states.

# Key Features of Enums: unique members, no duplicates; immutable; iterable; comparison (is or ==)
from enum import Enum

class TrafficLight(Enum):
    RED = "stop"
    YELLOW = "caution"
    GREEN = "go"

def handle_traffic_light(light):
    if light == TrafficLight.RED:
        print("Stop the car!")
    elif light == TrafficLight.YELLOW:
        print("Prepare to stop.")
    elif light == TrafficLight.GREEN:
        print("Go ahead.")

# Usage
handle_traffic_light(TrafficLight.GREEN)  # Output: Go ahead.


# Use enum.auto() to automatically assign values (starting from 1 and incrementing).
from enum import Enum, auto
class Animal(Enum):
    DOG = auto()  # 1
    CAT = auto()  # 2
    BIRD = auto() # 3

# Use the @unique decorator to ensure no duplicate values.
from enum import Enum, unique

@unique
class Day(Enum):
    MONDAY = 1
    TUESDAY = 2
    # WEDNESDAY = 1  # Raises ValueError due to duplicate value

# Enums can have methods for additional functionality.
from enum import Enum

class Mood(Enum):
    HAPPY = 1
    SAD = 2
    ANGRY = 3

    def describe(self):
        return f"Feeling {self.name.lower()} today!"

print(Mood.HAPPY.describe())  # Output: Feeling happy today!
