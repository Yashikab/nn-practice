# nn-practice

Just practice to run code with tensorflow on docker.

---

## Docker

Base Docker Image : yashikab/gputensorflow

Install pip pacakges in Dockerfile.

Run bash with docker-compose

```sh
# build from dockerfile.
$ docker-compose build ${service-name} # here "gpu or cpu"
# run
$ docker-compose run ${service-name} # here "gpu or cpu"
```

If you install something for general setting, you just write it on docker-manage repository and change the original docker image.

---

## Source

Using sample program: classification of mnist pictures. (python)

---

## Environment

tensorflow version : 1.13

cuda : 10.1

cudnn: 7
