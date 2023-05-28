#!/usr/bin/env bash

REPOS_DIR="data/repos"

main() {
    cd "$REPOS_DIR" || exit 1

    if [ ! -f "$REPO_LIST" ]; then
        echo "Repo list not found: $REPO_LIST"
        echo
        echo "Usage: $0 [--update|-u] <repo-list>"
        echo "  --update|-u: Update existing repos"
        echo "  <repo-list>: List of repos to clone"
        echo
        echo "By default, the repo list is repo-list.txt"
        exit 1
    fi

    while read -r remote; do
        if [[ "$remote" =~ ^# ]]; then
            continue
        fi

        clonedir=$(basename "$remote" | sed 's/\.git$//')
        if [ -d "$clonedir" ]; then
            if [ "$UPDATE" = true ]; then
                echo "Updating $remote..."
                git -C "$clonedir" pull
            fi
            continue
        fi

        echo "Cloning $remote..."
        git clone "$remote" || echo "Failed to get $remote"
    done < "$REPO_LIST"
    echo "Done."
}

if [ $# -eq 0 ]; then
    echo "Usage: $0 [--update|-u] <repo-list>"
    exit 1
fi

UPDATE=false
REPO_LIST="repo-list.txt"

while [ $# -gt 0 ]; do
    case "$1" in
        --update|-u)
            UPDATE=true
            shift
            ;;
        *)
            REPO_LIST=$(realpath "$1")
            shift
            ;;
    esac
    shift
done

main
