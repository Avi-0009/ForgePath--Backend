FROM golang:1.26-alpine

ENV CI=true \
    HOME=/tmp \
    GOPATH=/work/.go \
    GOCACHE=/work/.go-cache \
    GOTMPDIR=/work/.go-tmp \
    GOPROXY=off \
    GOTOOLCHAIN=local \
    GOENV=off \
    CGO_ENABLED=0 \
    GOFLAGS="-p=2"

WORKDIR /work

USER 65534:65534