Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

### Report Bugs

Report bugs at <https://github.com/gbouvignies/chemex/issues>.

If you are reporting a bug, please include:

-   Your operating system name and version.
-   Any details about your local setup that might be helpful in
    troubleshooting.
-   Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and
"help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with
"enhancement" and "help wanted" is open to whoever wants to implement
it.

### Write Documentation

ChemEx could always use more documentation, whether as part of the
official ChemEx docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at
<https://github.com/gbouvignies/chemex/issues>.

If you are proposing a feature:

-   Explain in detail how it would work.
-   Keep the scope as narrow as possible, to make it easier to
    implement.
-   Remember that this is a volunteer-driven project, and that
    contributions are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up PeakFit for local development.

1.  Fork the PeakFit repo on GitHub.

2.  Clone your fork locally:

        $ git clone git@github.com:your_name_here/PeakFit.git

3.  Install uv if you haven't already:

        $ curl -LsSf https://astral.sh/uv/install.sh | sh

4.  Set up your local development environment:

        $ cd PeakFit/
        $ uv sync --all-extras

5.  Create a branch for local development:

        $ git checkout -b name-of-your-bugfix-or-feature

    Now you can make your changes locally.

6.  When you're done making changes, check that your changes pass linting,
    type checking, and tests:

        $ uv run ruff check peakfit/
        $ uv run ruff format peakfit/
        $ uv run mypy peakfit/
        $ uv run pytest

7.  Commit your changes and push your branch to GitHub:

        $ git add .
        $ git commit -m "Your detailed description of your changes."
        $ git push origin name-of-your-bugfix-or-feature

8.  Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1.  The pull request should include tests.
2.  If the pull request adds functionality, the docs should be updated.
    Put your new functionality into a function with a docstring, and add
    the feature to the list in README.md.
3.  The pull request should work for Python 3.13+. Check GitHub Actions
    CI results and make sure that the tests pass.

Tips
----

To run a subset of tests:

    $ uv run pytest tests/unit/test_lineshapes.py

To run tests with coverage:

    $ uv run pytest --cov=peakfit --cov-report=html

To build the package locally:

    $ uv build
