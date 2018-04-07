import random
import string
import collections

def num_range(x):
    return([i for i in range(1,x)])

def square(x):
    return(x**2)

def cube(x):
    return(x**3)

def tesseract(x):
    return(x**4)

def sct(x, y, z):
    return(square(x) + cube(y) + tesseract(z))

def process_item(item):
    return {
        'name': item.name,
        'age': 2017 - item.born
    }

Scientist = collections.namedtuple('Scientist', [
    'name',
    'born',
])

scientists = (
    Scientist(name='Ada Lovelace', born=1815),
    Scientist(name='Emmy Noether', born=1882),
    Scientist(name='Marie Curie', born=1867),
    Scientist(name='Tu Youyou', born=1930),
    Scientist(name='Ada Yonath', born=1939),
    Scientist(name='Vera Rubin', born=1928),
    Scientist(name='Sally Ride', born=1951),
)


def rand_string(length, output):
    """ Generates a random string of numbers, lower- and uppercase chars. """
    rand_str = ''.join(random.choice(
                        string.ascii_lowercase
                        + string.ascii_uppercase
                        + string.digits)
                   for i in range(length))
    output.put(rand_str)
