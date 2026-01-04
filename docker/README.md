# Use Dockerfile.cuda to build image.

```bash
# You need to set FLAGSCALE_REPO, FLAGSCALE_BRANCH, and FLAGSCALE_COMMIT in Dockerfile.cuda to specify the FlagScale used in the installation environment
docker build -f Dockerfile.cuda -t flagscale:cuda12.4.1-cudnn9.5.0-python3.12-torch2.5.1-time2503251131 .
```

* Corresponding versions of `cuda` `cudnn` `python` `torch`: Can be manually specified in `docker/Dockerfile.cuda` and `install/install-requirements.sh` and `requirements`.
* `time`: Manually input the time of building the image, starting from year and accurate to minute, for example `25(year)03(month)25(day)11(hour)31(minute)`.

# Build an SSH login free image using existing images and Dockerfile.ssh.
NOTE:
   1. This construction method is not secure and is only for internal development use. Do not leak the built image.
   2. It is recommended to rebuild the image for different tasks each time to avoid environmental interference caused by the same key.

```bash
docker build --build-arg BASE_IMAGE=flagscale:cuda12.4.1-cudnn9.5.0-python3.12-torch2.5.1-time2503251131 --build-arg SSH_PORT=22 -f Dockerfile.ssh -t flagscale:cuda12.4.1-cudnn9.5.0-python3.12-torch2.5.1-time2503251131-ssh .
```

* `BASE_IMAGE`: Must be explicitly specified, enter the name of the base image used.
* `SSH_PORT`: Can be manually specified, default value is 22.
