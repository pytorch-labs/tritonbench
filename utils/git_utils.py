import subprocess


def checkout_submodules(repo_root):
    cmd = ["git", "submodule", "update", "--init", "--recursive"]
    subprocess.check_call(cmd, cwd=repo_root)
