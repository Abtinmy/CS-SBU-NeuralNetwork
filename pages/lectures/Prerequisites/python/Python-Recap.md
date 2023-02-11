---
layout: default
title: Python Recap
parent: Prerequisites
grand_parent: Lectures
nav_order: 0
has_children: false
permalink: /lectures/Prerequisites/python/Python-Recap
---





# Python Recap
**Author** : Mobin Nesari

**Prepared for** : The Artificial Neural Network Graduate course 2023 Shahid Beheshti University

## Introduction to Python
### Setting up the Environment
Before we dive into the Python language, it is important to ensure that we have the right environment set up.

To get started with Python, you will need to install Python and Jupyter Notebook.

You can download and install Python from the official website python.org. The latest version of Python as of this writing is Python 3.10.

To install Jupyter Notebook, you can use the following command in the terminal or command prompt:


```python
# >> pip install jupyter
```

### Basic Syntax
In this section, we will cover the basic syntax of Python. This includes:

- Data Types: Integer, Float, String, Boolean, etc.
- Variables: Naming conventions, assigning values to variables.
- Operators: Arithmetic, comparison, logical, and assignment operators.
- Loops: for and while loops.
- Functions: Defining and calling functions.

Let's start by creating our first Python program:


```python
print("Hello, World!")
```

    Hello, World!
    

This will display the text `Hello, World!` when you run the code.

## Intermediate Python

In this section, we will build on the basic syntax learned in the previous section. We will cover the following topics:

### Control Structures

Control structures are used to control the flow of execution in a program. Python provides two types of control structures:

- Conditional statements (`if-elif-else`): These statements allow you to execute different code blocks based on a set of conditions. The elif clause is used to check for multiple conditions. 

Here is an example:



```python
x = 10
if x > 0:
    print("x is positive")
elif x == 0:
    print("x is zero")
else:
    print("x is negative")

```

    x is positive
    

- Loops (`for` and `while`): Loops allow you to repeat a code block a certain number of times or until a specific condition is met. The while loop continues to execute its code block until the given condition is False.

Here is an example:


```python
i = 0
while i < 5:
    print(i)
    i += 1
```

    0
    1
    2
    3
    4
    

And here is an example of using `for` loop:


```python
for i in range(5):
    print(i)
```

    0
    1
    2
    3
    4
    

### List Comprehensions and Generator Expressions
List comprehensions and generator expressions are concise and readable ways to generate new lists or iterators in Python.

A list comprehension is a compact way to generate a new list from an existing list. Here is an example:


```python
numbers = [1, 2, 3, 4, 5]
squared_numbers = [x**2 for x in numbers]
print(squared_numbers) # Output: [1, 4, 9, 16, 25]
```

    [1, 4, 9, 16, 25]
    

A generator expression is similar to a list comprehension, but instead of creating a list, it creates a generator object. This is useful when working with large datasets that don't need to be stored in memory. Here is an example:


```python
numbers = (x**2 for x in range(10))
print(next(numbers)) # Output: 0
print(next(numbers)) # Output: 1
```

    0
    1
    

### Modules and Packages
Modules and packages are ways to organize and reuse code in Python.

A module is a file containing Python definitions and statements. You can import a module into another module or script using the `import` statement. Here is an example:


```python
# math_module.py
def square(x):
    return x**2

# main.py
import math_module
print(math_module.square(5)) # Output: 25

```

A package is a collection of modules. You can import modules from a package using the `from` keyword and the `import` statement. Here is an example:


```python
# package/
#   __init__.py
#   module1.py
#   module2.py

# main.py
from package import module1, module2

```

These are the basics of modules and packages in Python. You can learn more about them as you continue to use Python.

## Object Oriented Programming (OOP)
Object-oriented programming (OOP) is a programming paradigm based on the concept of objects, which can contain data and code that manipulates that data. In Python, everything is an object, including functions, modules, and classes. In this section, we'll focus on classes and how to use them to build objects.

### Class Definition
A class is a blueprint for creating objects. It defines a set of attributes and methods that objects created from the class will have. Here's an example of a class definition for a simple point in 2D space:


```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, other_point):
        return ((self.x - other_point.x)**2 + (self.y - other_point.y)**2)**0.5
```

In this example, `Point` is the class name, and `__init__` and `distance` are two methods. The `__init__` method is a special method called a constructor that is automatically called when a new Point object is created. The `self` parameter is a reference to the object being created and is required for all methods in a class.

### Instantiating Objects
Once you have defined a class, you can create objects from it by calling the class as if it were a function. Here's an example:


```python
p1 = Point(0, 0)
p2 = Point(3, 4)

print(p1.distance(p2)) # Output: 5.0
```

    5.0
    

In this example, `p1` and `p2` are instances of the `Point` class, and they each have their own `x` and `y` attributes and can call the `distance` method.

### Inheritance
Inheritance is a mechanism that allows you to create a new class that is a modified version of an existing class. The new class is called a subclass, and the existing class is called a superclass. A subclass inherits all of the attributes and methods of its superclass. Here's an example:


```python
class ColoredPoint(Point):
    def __init__(self, x, y, color):
        super().__init__(x, y)
        self.color = color

cp = ColoredPoint(0, 0, 'red')
print(cp.distance(Point(3, 4))) # Output: 5.0
print(cp.color) # Output: 'red'
```

    5.0
    red
    

In this example, `ColoredPoint` is a subclass of `Point`, so it inherits the `distance` method from `Point`. It also has a new `color` attribute and a modified `__init__` method that calls the `__init__` method of the superclass using `super().__init__(x, y)` and then adds the new `color` attribute.

### Polymorphism
Polymorphism is the ability of objects of different classes to be used interchangeably as long as they implement the same methods. Here's an example:


```python
class Circle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r
    def area(self):
        return 3.14 * self.r**2

    def distance(self, other_shape):
        return ((self.x - other_shape.x)**2 + (self.y - other_shape.y)**2)**0.5
    
c = Circle(0, 0, 5)
p = Point(3, 4)

shapes = [c, p]

for shape in shapes:
    print(shape.distance(Point(0, 0))) # Output: 5.0 and 5.0
    print(shape.area()) # Output: 25.13 for Circle and error for Point
```

    0.0
    78.5
    5.0
    


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Input In [11], in <cell line: 17>()
         17 for shape in shapes:
         18     print(shape.distance(Point(0, 0))) # Output: 5.0 and 5.0
    ---> 19     print(shape.area())
    

    AttributeError: 'Point' object has no attribute 'area'



In this example, both the `Circle` and `Point` classes have a `distance` method, so they can be used interchangeably in the `shapes` list. However, only the `Circle` class has an `area` method, so trying to call `area` on a `Point` object will result in an error.

This is the basic of Object Oriented Programming in Python. With these concepts, you can write more advanced, efficient, and organized code in Python.


## Data Structures in Python
In this section, you will learn about the various data structures available in Python and how to use them. Some of the common data structures in Python are:
- Lists
- Tuples
- Dictionaries
- Sets

### Lists
A list is an ordered, mutable, and heterogeneous collection of items. It is defined by square brackets `[]` and items are separated by commas `,`.


Here is an example:


```python
fruits = ['apple', 'banana', 'cherry']
print(fruits[0]) # Output: 'apple'
fruits[0] = 'mango'
print(fruits[0]) # Output: 'mango'
```

    apple
    mango
    

In the above example, `fruits` is a list of three items. Lists are 0-indexed in Python, which means the first item has an index of 0, the second item has an index of 1, and so on.


### Lists and Built-in Methods
Lists are one of the most important data structures in Python and it provides several built-in methods to manipulate the list. Here, we will go over some of the most common built-in methods for lists:

#### append()
The `append()` method is used to add an element to the end of the list.


```python
fruits = ['apple', 'banana', 'cherry']
fruits.append('orange')
print(fruits)
# Output: ['apple', 'banana', 'cherry', 'orange']
```

    ['apple', 'banana', 'cherry', 'orange']
    

#### extend()
The `extend()` method is used to add multiple elements to the end of the list.


```python
fruits = ['apple', 'banana', 'cherry']
new_fruits = ['orange', 'grapes']
fruits.extend(new_fruits)
print(fruits)
# Output: ['apple', 'banana', 'cherry', 'orange', 'grapes']
```

    ['apple', 'banana', 'cherry', 'orange', 'grapes']
    

#### remove()
The `remove()` method is used to remove an element from the list. It removes the first occurrence of the element.


```python
fruits = ['apple', 'banana', 'cherry']
fruits.remove('banana')
print(fruits)
# Output: ['apple', 'cherry']
```

    ['apple', 'cherry']
    

#### sort()
The `sort()` method is used to sort the elements of the list in ascending order.


```python
fruits = ['cherry', 'banana', 'apple']
fruits.sort()
print(fruits)
# Output: ['apple', 'banana', 'cherry']
```

    ['apple', 'banana', 'cherry']
    

#### pop()
The `pop()` method is used to remove and return the last element of the list.


```python
fruits = ['apple', 'banana', 'cherry']
last_fruit = fruits.pop()
print(fruits)
# Output: ['apple', 'banana']
print(last_fruit)
# Output: 'cherry'
```

    ['apple', 'banana']
    cherry
    

### Tuples
A tuple is similar to a list, but it is an ordered, immutable, and heterogeneous collection of items. It is defined by parentheses `()` and items are separated by commas `,`.

Here is an example:


```python
fruits = ('apple', 'banana', 'cherry')
print(fruits[0]) # Output: 'apple'
```

    apple
    

In the above example, `fruits` is a tuple of three items. Tuples are also 0-indexed in Python, but unlike lists, they cannot be modified after creation.

### Dictionaries
A dictionary is an unordered collection of key-value pairs. It is defined by curly braces `{}` and items are separated by colons `:`.

Here is an example:


```python
fruits = {'apple': 100, 'banana': 200, 'cherry': 300}
print(fruits['apple']) # Output: 100
fruits['apple'] = 150
print(fruits['apple']) # Output: 150
```

    100
    150
    

In the above example, `fruits` is a dictionary with three key-value pairs. Dictionaries are similar to lists in terms of indexing, but instead of an integer index, you use a key to access the value.

### Sets
A set is an unordered collection of unique items. It is defined by curly braces `{}` or the `set` function.

Here is an example:


```python
fruits = {'apple', 'banana', 'cherry'}
print('apple' in fruits) # Output: True
fruits.add('mango')
print(fruits) # Output: {'banana', 'cherry', 'apple', 'mango'}
```

    True
    {'banana', 'cherry', 'mango', 'apple'}
    

In the above example, `fruits` is a set with three items. Sets are similar to lists and tuples in terms of iteration and membership testing, but unlike lists and tuples, sets do not allow duplicates.

These are the most commonly used data structures in Python. Understanding these data structures and knowing how to use them is a crucial part of becoming a proficient Python programmer.

## File Handling in Python
In this section, you will learn how to read and write files in Python. Python provides several ways to work with files, including the built-in `open` function and various module functions in the `os` and `os.path` modules.

### Reading Files
The most basic way to read a file is to use the `open` function, which returns a file object that you can use to read the contents of the file.

Here is an example:


```python
with open('example.txt', 'r') as file:
    content = file.read()
    print(content)
```

    Hello, World !
    

In the above example, the open function is used to open the `example.txt` file in read mode (`'r'`). The `with` statement is used to ensure that the file is properly closed after it is read, even if an exception occurs. The `read` method is used to read the contents of the file and store it in the `content` variable.

### Writing Files
To write to a file, you can use the `open` function in write mode (`'w'`) and write to the file using the `write` method.

Here is an example:


```python
with open('example.txt', 'w') as file:
    file.write('Hello, world2!')
```

In the above example, the `open` function is used to open the `example.txt` file in write mode (`'w'`). The `with` statement is used to ensure that the file is properly closed after it is written, even if an exception occurs. The `write` method is used to write the string `'Hello, world!'` to the file.

### Appending to Files
To append to a file, you can use the `open` function in append mode (`'a'`) and write to the file using the `write` method.

Here is an example:


```python
with open('example.txt', 'a') as file:
    file.write('\nThis is a new line.')
```

In the above example, the `open` function is used to open the `example.txt` file in append mode (`'a'`). The `with` statement is used to ensure that the file is properly closed after it is written, even if an exception occurs. The `write` method is used to append the string `'\nThis is a new line.'` to the file.

These are the basic ways to read and write files in Python. In practice, you may also need to work with other file formats, such as CSV and JSON, which require specialized libraries and techniques.

## Packages and Library Management in Python
In this section, you will learn how to install packages in Python using the `pip` package manager. The `pip` package manager is included with the standard Python distribution and is the most common way to install packages in Python.

### Installing a Package
To install a package in Python using `pip`, you can use the following command in your terminal or command prompt:


```python
# >> pip install [package_name]
```

For example, to install the `numpy` package, you would use the following command in your terminal or command prompt:


```python
# >> pip install numpy
```

### Installing a Specific Version of a Package
You can also install a specific version of a package by including the version number in the `pip` command. For example, to install version 1.16.0 of the `numpy` package, you would use the following command:


```python
# >> pip install numpy==1.16.0
```

### Installing a Package from a Requirements File
A requirements file is a text file that lists the packages and their versions required for a project. To install the packages listed in a requirements file, use the following command in your terminal or command prompt:


```python
# >> pip install -r requirements.txt
```

where `requirements.txt` is the name of your requirements file.

### Importing Packages
You can import a package in Python using the `import` statement. For example, to import the `math` package, use the following code:


```python
import math
```

### Using Packages
Once a package is imported, you can use the functions and classes it provides in your code. For example, the `math` package provides the `sqrt` function, which calculates the square root of a number:


```python
import math

result = math.sqrt(16)
print(result) # 4.0
```

    4.0
    

### Managing Packages
You can manage packages in Python using the `pip` package manager. To list the packages installed in your environment, use the following command:


```python
# >> pip list
```

To update a package, use the following command:


```python
# >> pip install --upgrade [package_name]
```

To uninstall a package, use the following command:


```python
# >> pip uninstall [package_name]
```

Installing packages in Python is easy and straightforward using the `pip` package manager. By installing packages, you can leverage the work of others and save time and effort in your own projects. With the knowledge of how to install packages in Python, you can start using powerful libraries and packages in your projects and take your development skills to the next level.

## Advanced Topics in Python
In this section, you will learn about some advanced topics in Python, including error handling, decorators, generators, and context managers. These topics are useful for building robust and scalable applications in Python.

### Error Handling
Error handling is a critical aspect of programming in any language. In Python, you can use try-except blocks to handle exceptions that might occur in your code.

Here is an example:


```python
try:
    # code that might raise an exception
    result = 1 / 0
except ZeroDivisionError as error:
    print(f'An error occurred: {error}')
```

    An error occurred: division by zero
    

In the above example, a `ZeroDivisionError` exception is raised when attempting to divide by zero. The `except` block is used to catch the exception and print an error message.

### Decorators
Decorators are a powerful feature in Python that allow you to modify the behavior of functions and classes. Decorators are applied to functions or classes using the `@` symbol.

Here is an example:


```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print('Before the function is called.')
        result = func(*args, **kwargs)
        print('After the function is called.')
        return result
    return wrapper

@my_decorator
def my_function(a, b):
    return a + b
```

In the above example, the `my_decorator` function is defined as a decorator. The `my_function` function is decorated with the `my_decorator` decorator using the `@` symbol. When the `my_function` is called, the behavior of the function is modified by the `my_decorator` to print messages before and after the function is called.



### Generators
Generators are a convenient way to generate a sequence of values in Python. Generators are defined using the `yield` statement.

Here is an example:



```python
def my_generator():
    for i in range(3):
        yield i

for item in my_generator():
    print(item)
```

    0
    1
    2
    

In the above example, the `my_generator` function is defined as a generator. The `yield` statement is used to generate a sequence of values. The `for` loop is used to iterate over the values generated by the generator.

### Context Managers
Context managers are a convenient way to manage resources in Python. Context managers are defined using the with statement.

Here is an example:



```python
class File:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

with File('example.txt', 'w') as file:
    file.write('Hello, world!')
```

In the above example, the `File` class is defined as a context manager. The `__enter__` method is used to open the file and the `__exit__` method is used to close the file. The `with` statement is used to manage


```python

```
