# Release a New Version

PyPi Instructions: https://packaging.python.org/distributing/#uploading-your-project-to-pypi

Twin command-line tool for registering and uploading packages: https://github.com/pypa/twine


Commit code edits to GitHub after making all your awesome changes.  Update version
numbers in setup.py, version.py, etc.  Making sure to also remove 'dev' descriptor if you use
one (https://packaging.python.org/tutorials/distributing-packages/#choosing-a-versioning-scheme).

```bash
git add <any new stuff>
git commit -a
```

Create source and binary distribution files.  Twine will handle registering the project if this is
the first time.


```bash
rm -rf dist

python setup.py sdist bdist_wheel

twine upload dist/*

python setup.py clean
```


After the above it's time to go back to developing the next great release.  Update current version
numbers by adding 'dev' (if that's your style) and increment the minor number. Commit this change
and then get back to work.


```bash
git commit -a
```
