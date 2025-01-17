import subprocess
from tritonbench.utils.path_utils import REPO_PATH


def get_branch(repo, commit) -> str:
    """Get branch name with a given commit in a git repo.
    repo: URL or local git repo path
    commit: hash of a commit
    If a commit does not belong to any branch, return "unknown"
    """
    pass

def get_commit_time(repo, commit) -> str:
    """Get commit time in a git repo.
    repo: URL or local git repo path
    commit: hash of a commit
    If a commit does not exist, return "unknown" 
    """
    pass

def get_current_hash(repo: str) -> str:
    """Get the HEAD hash of a git repo.
    repo: local git repo path"""
    try:
        cmd = ["git", "rev-parse", "--verify", "HEAD"]
        output = subprocess.check_output(cmd, cwd=REPO_PATH).decode().strip()
        return output
    except subprocess.SubprocessError:
        return "unknown"
