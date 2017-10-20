from setuptools import setup, find_packages

setup(
    name='daria',
    version='0.0.1',
    description='pytorch trainer',
    author='odanado',
    author_email='odan3240@gmail.com',
    url='https://github.com/odanado/daria',
    license='MIT License',
    packages=find_packages(exclude=['tests', 'tests.*']),
    install_requires=['slackweb'],
    tests_require=['mock'],
    test_suite='tests',
)
