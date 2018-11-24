# ImageAligner
Python based service to find sheets within a picture, cut them out and align them

## Run as daemon
To run the server as a daemon, copy `imagealigner.service` to `~/.config/systemd/imagealigner.service`. Make sure the path(under "Service" - "ExecStart") in the file points to the right path. Then start the service with:
```sh
systemd --user start detection_service
```
Make the service autostart with:
```sh
systemd --user enable detection_service
```
