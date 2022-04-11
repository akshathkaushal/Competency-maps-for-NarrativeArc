# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/WSL-IIITB/CompetencyMaps/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

competency_maps could always use more documentation, whether as part of the
official competency_maps docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/WSL-IIITB/CompetencyMaps/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)
  
### Other help

You can contribute by spreading a word about this library.
It would also be a huge contribution to write
a short article on how you are using this project.
You can also share your best practices with us.


## How to contribute

### Dependencies

We use `poetry` to manage the [dependencies](https://github.com/python-poetry/poetry).
If you dont have `poetry` installed, you should run the command below.

```bash
make download-poetry
```

To install dependencies and prepare [`pre-commit`](https://pre-commit.com/) hooks you would need to run `install` command:

```bash
make install
```

To activate your `virtualenv` run `poetry shell`.

### Codestyle

After you run `make install` you can execute the automatic code formatting.

```bash
make codestyle
```

#### Checks

Many checks are configured for this project. Command `make check-style` will run black diffs, darglint docstring style and mypy.
The `make check-safety` command will look at the security of your code.

You can also use `STRICT=1` flag to make the check be strict.

#### Before submitting

Before submitting your code please do the following steps:

1. Add any changes you want
1. Add tests for the new changes
1. Edit documentation if you have changed something significant
1. Run `make codestyle` to format your changes.
1. Run `STRICT=1 make check-style` to ensure that types and docs are correct
1. Run `STRICT=1 make check-safety` to ensure that security of your code is correct
