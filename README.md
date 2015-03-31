# pug-invest

[![Build Status](https://travis-ci.org/hobson/pug-invest.svg?branch=master "Travis Build & Test Status")](https://travis-ci.org/hobson/pug-invest)
[![Coverage Status](https://coveralls.io/repos/hobson/pug-invest/badge.png)](https://coveralls.io/r/hobson/pug-invest)
[![Version Status](https://pypip.in/v/pug-invest/badge.png)](https://pypi.python.org/pypi/pug-invest/)
[![Downloads](https://pypip.in/d/pug-invest/badge.png)](https://pypi.python.org/pypi/pug-invest/)

## Investing (finance) machine learning algorithms 

This sub-package of the pug namespace package, provides utilities and algorithms for manipulating and predictin financial time-series data.

---

## Installation

### On a Posix System

You really want to contribute, right?

    git clone https://github.com/hobson/pug-invest.git

If your a user and not a developer, and have an up-to-date posix OS with the postgres, xml2, and xlst development packages installed, then just use `pip`.

    pip install pug-invest

### Fedora

If you're on Fedora >= 16 but haven't done a lot of python binding development, then you'll need some libraries before pip will succeed.

    sudo yum install -y python-devel libxml2-devel libxslt-devel gcc-gfortran python-scikit-learn postgresql postgresql-server postgresql-libs postgresql-devel
    pip install pug

### Bleeding Edge

Even the releases are very unstable, but if you want to have the latest, most broken code

    pip install git+git://github.com/hobsonlane/pug.git@master

### Warning

This software is in alpha testing.  Install at your own risk.

---

## Development

I love merging PRs and adding contributors to the `__authors__` list:

    git clone https://github.com/hobson/pug-invest.git


