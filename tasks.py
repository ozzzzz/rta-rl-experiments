import subprocess

import invoke


def current_git_sha():
    """Get the current sha from git cli using subprocess."""
    try:
        rev_parse_out = (
            subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                stderr=subprocess.STDOUT,
            )
            .decode()
            .strip()
            .split("\n")
        )
    except subprocess.CalledProcessError as e:
        msg = "Could not get current commit sha."
        raise invoke.exceptions.Exit(msg) from e

    return rev_parse_out[-1]


def current_os() -> str:
    """Get the current os from git cli using subprocess."""
    try:
        rev_parse_out = (
            subprocess.check_output(
                [
                    "uname",
                    "-s",
                ],
                stderr=subprocess.STDOUT,
            )
            .decode()
            .strip()
            .split("\n")
        )
    except subprocess.CalledProcessError as e:
        msg = "Could not get current os."
        raise invoke.exceptions.Exit(msg) from e

    return rev_parse_out[-1].lower()


def current_branch():
    """Get the current branch from git cli using subprocess."""
    try:
        rev_parse_out = (
            subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "--tags",
                    "--abbrev-ref",
                    "HEAD",
                ],
                stderr=subprocess.STDOUT,
            )
            .decode()
            .strip()
            .split("\n")
        )
    except subprocess.CalledProcessError as e:
        msg = "Could not get current branch name."
        raise invoke.exceptions.Exit(msg) from e

    return rev_parse_out[-1]


def enforce_branch(branch_name):
    """Enforce that the current branch matches the supplied branch_name."""
    if current_branch() != branch_name:
        msg = f"Command can not be run outside of {branch_name}."
        raise invoke.exceptions.Exit(msg)


@invoke.task
def install(context):
    """Install production requirements."""
    context.run("poetry install --only main")


@invoke.task
def install_dev(context):
    """Install development requirements."""
    context.run("poetry install")
    context.run("poetry run pre-commit install")
    context.run(
        """
        echo "Generating pyrightconfig.json";
        echo "{\\"venv\\": \\".\\", \\"venvPath\\": \\"$(poetry env info -p)\\", \\"exclude\\": [\\"tests\\"], \\"include\\": [\\"rta_rl\\"]}" > pyrightconfig.json
    """,
    )


@invoke.task
def check_style(context):
    """Run style checks."""
    context.run("ruff .")


@invoke.task
def tests(context):
    """Run pytest unit tests."""
    context.run("pytest -x -s")


@invoke.task
def tests_coverage(context, output="xml"):
    """Run pytest unit tests with coverage."""
    context.run(f"pytest --cov=rta_service -x --cov-report={output}")


@invoke.task
def docker_build(context):
    """Build docker image."""
    git_sha = current_git_sha()
    sed_options = "-i ''" if current_os() == "darwin" else "-i"
    sed_command = f"sed {sed_options} 's/__version__.*/__version__ = \"{git_sha}\"/' rta_service/__init__.py"
    context.run(sed_command)
    context.run(f"yq -i '.image.tag = \"{git_sha}\"' chart/values.yaml")
    # TODO: define the docker registry
    context.run(f"docker build -t  TODO/rta_service:{git_sha} .")


@invoke.task
def post_docker_build(context):
    """Post docker build steps."""
    context.run(
        "git add . && git commit -a -m 'up version' && git push origin master:master",
    )
