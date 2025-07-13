

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

"""

