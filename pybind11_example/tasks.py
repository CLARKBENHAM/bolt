
import invoke
from invoke import task


@task
def build_ex(c):
    """how to avoid specify all build files?"""
    invoke.run(
        "g++ -O3 -Wall -Werror -shared -std=c++11 -fPIC "
        "`python3 -m pybind11 --includes` "
        "-I /usr/include/python3.7 -I .  "
        "{0} add.cpp "
        "-o {1}`python3.7-config --extension-suffix` "
        "-L. -Wl,-rpath,.".format('bind.cpp', 'example')
    )
    print('ran')
