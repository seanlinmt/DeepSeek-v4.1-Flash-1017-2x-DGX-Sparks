# Engine branches (git bundles)

`structured-output.bundle` carries the engine branch `structured-output`: every commit of it,
including the `so-base` commit it was cut from. It exists because the engine repo is
`github.com/bertholomus/TensorFold`, which this project has no write access to, and the patch
`zig/patches/0034-structured-output.patch` reproduces the tree but not the history.

Restore on a fresh repository:

    git init engine && cd engine
    git fetch https://github.com/bertholomus/TensorFold.git 629c524a0ea57512a5342abb498d1b7c25353343
    git fetch /path/to/structured-output.bundle structured-output:structured-output
    git checkout structured-output

The public pin `629c524` is the bundle's only prerequisite, so nothing else has to exist first.
Verified: a fresh repo with only that fetch restores the branch at its tip.

The engine branch is also pushed to the fork **seanlinmt/TensorFold** (branch `structured-output`,
tag `so-base`), which is the canonical remote; the bundle here is a network-free backup.
