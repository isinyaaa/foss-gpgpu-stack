#!/usr/bin/env bash

repo_list="$1"

if [ -z "$repo_list" ]; then
    echo "Usage: $0 <repo-list>"
    exit 1
fi

test -d repos || {
    mkdir repos || exit 1
}

while read repo; do
    if [[ "$repo" =~ ^# ]]; then
        continue
    fi

    # IS_GITFORGE=false
    # for i in {"github","gitlab","bitbucket"}; do
    #     if [[ "$repo" =~ "$i" ]]; then
    #         IS_GITFORGE=true
    #         break
    #     fi
    # done

    foldername=$(basename "$repo" | sed 's/\.git$//')

    if [ -d "repos/$foldername" ]; then
        if [ "$2" = "--update" ]; then
            echo "Updating $repo..."
            (
                cd "repos/$foldername"
                git pull
            )
        fi
        continue
    fi

    echo "Cloning $repo..."
    (
        cd repos
        git clone "$repo" || {
            echo "Failed to get $repo"
            exit 1
        }
    )
done < "$repo_list"
echo "Done."
