# Engine branches (git bundles)

`structured-output.bundle` carries the engine branch `structured-output`, cut from `so-base`:
the commits behind `zig/patches/0034-structured-output.patch`. The patch reproduces the tree;
this bundle preserves the commit history. It exists because the engine repo is
`github.com/bertholomus/TensorFold`, which this project has no write access to.

Restore on a fresh clone of the fork:

    git fetch https://github.com/bertholomus/TensorFold.git 629c524a0ea57512a5342abb498d1b7c25353343
    git checkout -B so-base FETCH_HEAD
    for p in zig/patches/00*.patch; do case "$(basename "$p")" in 0034-*) ;; *) git apply "$p";; esac; done
    git add -A && git commit -m "base: TF_REF + patches before 0034"
    git tag so-base
    git fetch /path/to/structured-output.bundle structured-output:structured-output
    git checkout structured-output

`so-base` is the engine pin plus every hand-written patch except `0034`; the bundle is relative
to it, so the two together reproduce the branch exactly.
