
# Contributing to OpenWillis

Welcome to the OpenWillis project! We’re thrilled that you’re interested in contributing to our community. Your contributions are essential to help us solve the measurement problem in health and build trust in these measurements. Whether you’re here to fix a bug, add a new digital measure, or improve the documentation, your help is highly appreciated. We’re eager to collaborate with you and excited to see what we can achieve together. Thank you for considering contributing to OpenWillis!


## Table of Contents



- [Contributing to OpenWillis](#contributing-to-openwillis)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Set Up Local Development Environment](#set-up-local-development-environment)
    - [What to Know Before You Get Started](#what-to-know-before-you-get-started)
  - [How to Contribute](#how-to-contribute)
    - [Reporting Bugs](#reporting-bugs)
    - [Suggesting Enhancements](#suggesting-enhancements)
    - [Your First Code Contribution](#your-first-code-contribution)
  - [Pull Requests](#pull-requests)
    - [Styleguides](#styleguides)
      - [Git Commit Messages](#git-commit-messages)
      - [Python Styleguide](#python-styleguide)
      - [Documentation Styleguide](#documentation-styleguide)
  - [Community and Support](#community-and-support)
    - [How to Get Help](#how-to-get-help)
    - [Contributing Beyond Code](#contributing-beyond-code)


## Getting Started

Contributing to OpenWillis can be a rewarding experience. To ensure a smooth contribution process for everyone involved, we’ve outlined some steps and guidelines you should be aware of before you start contributing.


### Set Up Local Development Environment

To start contributing to OpenWillis, you need to set up your local development environment. You can find the instructions in the [Getting started](https://github.com/bklynhlth/openwillis/wiki/Getting-started) wiki page. After that make sure to read through the [List of functions](https://github.com/bklynhlth/openwillis/wiki/2.-List-of-functions) wiki page to understand the different functions and their purpose and the [Research Guidelines](https://github.com/bklynhlth/openwillis/wiki/3.-Research-guidelines) wiki page to gain a deeper understanding on how the input data should be structured and collected.


### What to Know Before You Get Started

Before you begin contributing to OpenWillis, there are a few things you should know:



* **Project Overview**: Familiarize yourself with the project’s goals, architecture, and development roadmap. This information can be found in the README file and the project wiki.
* **Issue Tracking**: We use GitHub issues to track bugs and feature requests. Before creating a new issue, please check to ensure it hasn’t already been reported. If you find an existing issue that matches your own, add a comment to let us know you’re experiencing the same problem.
* **Contribution Process**: Contributions to OpenWillis should be made via pull requests to the latest version’s branch and not directly to main. Each pull request should be linked to an open issue. If there isn’t an issue that matches your contribution, please create one before submitting your pull request.

Getting involved in an open-source project like OpenWillis is not only about submitting code. There are many ways to contribute:



* **Reporting Bugs**: If you find a bug, report it via GitHub issues. Include as much detail as possible to help us understand and replicate the issue.
* **Suggesting Enhancements**: Have an idea for making OpenWillis better? We’d love to hear it! Create an issue to suggest enhancements.
* **Improving Documentation**: Good documentation is crucial for any project. If you see areas that could be improved or clarified, don’t hesitate to contribute or ask for clarifications.

By following these guidelines, you’ll be well on your way to contributing to OpenWillis.


## How to Contribute

Contributing to OpenWillis involves a range of activities from reporting bugs, suggesting enhancements, to contributing code. Here’s how you can contribute:


### Reporting Bugs

Bugs are tracked as GitHub issues. If you find a bug in the project, please check the existing issues to ensure it hasn’t been reported already. If it’s a new issue, open a new issue and include:



* A clear and descriptive title.
* A detailed description of the bug. Include steps to reproduce the issue, the expected outcome, and the actual outcome.
* Any relevant information such as error messages, screenshots, or system information.


### Suggesting Enhancements

Enhancements and feature requests are also tracked as GitHub issues. Before suggesting an enhancement, please check if it has already been suggested. If not, open a new issue and include:



* A clear and descriptive title.
* A detailed explanation of the proposed feature or enhancement. Include any specific use cases or benefits.
* Any preliminary ideas on how to implement the enhancement, if you have them.


### Your First Code Contribution

Unsure where to start contributing code? Look for issues labeled `good first issue` or `help wanted`. These issues are well-suited for new contributors. Here are some steps to follow:



1. **Fork the repository**: Create your own fork of the project to work on your contribution.
2. **Clone your fork**: Clone your fork to your local machine and set up the development environment.
3. **Create a branch**: Always make your changes in a new branch specific to the issue you’re addressing.
4. **Make your changes**: Work on the issue as described, adhering to the project’s coding standards and guidelines.
5. **Commit your changes**: Use clear and meaningful commit messages. Follow any commit message guidelines the project has.
6. **Push your changes**: Push your changes to your fork on GitHub.
7. **Submit a Pull Request (PR)**: Open a pull request from your fork to the latest version branch of the original repository. Provide a clear and detailed title and description of your changes. Link to the issue your PR addresses.

For more detailed instructions on how to prepare your pull request, see [Pull Requests](#bookmark=id.2s8eyo1).

Contributing to OpenWillis is not just about code. You can contribute in many ways:



* **Improving documentation**
* **Reviewing pull requests**
* **Participating in discussions on issues**

Every contribution is valuable and appreciated.


## Pull Requests

The process for submitting a pull request is as follows:



1. **Link the PR to an Issue**: Before submitting a PR, make sure it’s linked to an issue. If the PR addresses an issue not yet reported, please create an issue first.
2. **Do Not Submit Drafts**: Only submit PRs that are complete and ready for review. Drafts or work-in-progress PRs should be marked as such.
3. **Follow the Style Guides**: Ensure your code adheres to the style guides outlined below.
4. **Pass All Checks**: Your PR should pass all the checks and tests set up by the project. This may include unit tests, linting, and other CI/CD processes.
5. **Review Process**: Once submitted, your PR will be reviewed by the project maintainers. Be open to feedback and ready to make changes as requested.
6. **Merge**: After your PR has been approved, a project maintainer will merge it into the project.


### Styleguides


#### Git Commit Messages



* Use the present tense (“Add feature” not “Added feature”).
* Use the imperative mood (“Move cursor to…” not “Moves cursor to…”).
* Limit the first line to 72 characters or less.
* Reference issues and pull requests liberally after the first line.
* Consider starting the commit message with an applicable emoji to categorize your changes:
    * 📝 `:memo:` when writing docs,
    * 🐛 `:bug:` when fixing a bug,
    * ✨ `:sparkles:` when introducing new features,
    * 🚧 `:construction:` when working on WIP code.


#### Python Styleguide



* Follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code.
* Use descriptive variable names and keep function definitions clear and concise.
* Include comments in your code where necessary to explain complex logic.


#### Documentation Styleguide



* Use Markdown for documentation.
* Keep documentation up-to-date with code changes.
* Provide examples and usage scenarios for complex functionalities.
* Ensure clarity and readability for a diverse audience.


## Community and Support


### How to Get Help

If you need help or have questions, here are a few ways to get support:



* **GitHub Issues**: For detailed questions or issues with the codebase, open a new issue.
* **Discussions**: For general questions and discussions, use the project’s [Discussions](https://github.com/bklynhlth/openwillis/discussions) section.
* **OpenWillis user community**: Join our user community. We use the listserv to keep users in the loop on new measures and updates to existing ones. You’ll also be invited to any workshops or user meetups we organize throughout the year. [Join the OpenWillis user community](https://docs.google.com/forms/d/e/1FAIpQLScwc7IKChfJ1Wn373Ar3zjZmxQNIVK30OGEVnrHGOjW0ie1Sg/viewform?usp=sf_link).


### Contributing Beyond Code

OpenWillis is more than just code. Here’s how you can contribute in other ways:



* **Participate in Community Events**: Join workshops, meetups, or webinars to connect with other contributors and users of the platform.
* **Feedback**: Provide feedback on features, usability, and documentation.
* **Spread the Word**: Share OpenWillis with others who might benefit from it.

Thank you for contributing to OpenWillis! Your efforts help build a vibrant and supportive community around this project. We’re excited to work with you and see the impact of your contributions!
